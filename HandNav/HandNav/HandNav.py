import time

import vtk
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from vtk.util import numpy_support
import threading
import enum

import os
import numpy as np
import importlib

import Resources
import Resources.Logic.utils as utils

importlib.reload(Resources.Logic.utils)


#
# HandNav
#

class HandNav(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "HandNav"
        self.parent.categories = ["Informatics"]
        self.parent.dependencies = ["Markups"]
        self.parent.contributors = ["Étienne Léger (BWH), Fryderyk Kögl (TUM, BWH), Nazim Haouchine (BWH)"]
        self.parent.helpText = """
    Trackerless craniotomy planning.
    https://github.com/koegl/HandNav
    """
        self.parent.acknowledgementText = """
    This extension was developed at the Brigham and Women's Hospital by Étienne Léger, Fryderyk Kögl, Nazim Haouchine.

    This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
    and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
    """


#
# HandNavWidget
#
class HandNavWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    TRACKED_OBJECT_NAME = "OpenIGTLink"
    MAX_NUMBER_OF_CONTROL_POINTS = 100
    PORT_NUMBER = 18945
    NUMBER_OF_POINTS_FOR_AVG = 100
    TRACE_SMOOTHING = 20

    def __init__(self, parent=None):
        """
    Called when the user opens the module the first time and the widget is initialized.
    """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

        self.curve_nodes = {}

        self.world_landmarks = None

        self.previous = None

        self.points = None  
        
        self.point_recording_array = []  # array used to store points during recording

        # empty list of length 4
        self.recorded_points = [None for _ in range(self.MAX_NUMBER_OF_CONTROL_POINTS)]
        self.recorded_points_stds = [None for _ in range(self.MAX_NUMBER_OF_CONTROL_POINTS)]

        self.recorded_points_labels = []

        self.point_append_observer = None  # observer that is attached to the IGTLink node

        self.progress_bar_observer = None
        self.progress_bar = None

        self.current_point = 0

        self.stored_points_node = None

        self.trace_curves = [None for _ in range(self.MAX_NUMBER_OF_CONTROL_POINTS)]
        self.trace_curve_counter = 0

        self.record = False
        self.trace = False

        self.recording_finger = Finger.index * 4

    def setup(self):
        """
    Called when the user opens the module the first time and the widget is initialized.
    """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/HandNav.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = HandNavLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).

        # Buttons
        self.ui.initialiseButton.connect('clicked(bool)', self.onInitialiseButton)
        self.ui.connectObserverButton.connect('clicked(bool)', self.onConnectObserverButton)
        # parameters
        self.ui.thumbButton.connect('clicked(bool)', self.onThumbButton)
        self.ui.indexButton.connect('clicked(bool)', self.onIndexButton)
        self.ui.middleButton.connect('clicked(bool)', self.onMiddleButton)
        self.ui.ringButton.connect('clicked(bool)', self.onRingButton)
        self.ui.pinkyButton.connect('clicked(bool)', self.onPinkyButton)
        self.recording_finger = Finger.index * 4
        self.ui.indexButton.enabled = False  # it's the default, so disable it

        # point recording
        self.ui.recordPointButton.connect('clicked(bool)', self.onRecordPointButton)
        self.ui.deletePointButton.connect('clicked(bool)', self.onDeletePointButton)
        # tracing
        self.ui.startTraceButton.connect('clicked(bool)', self.onStartTraceButton)
        self.ui.stopTraceButton.connect('clicked(bool)', self.onStopTraceButton)
        self.ui.deleteTracesButton.connect('clicked(bool)', self.onDeleteTracesButton)
        # results
        self.ui.printResultsButton.connect('clicked(bool)', self.onPrintResultsButton)
        # evaluation
        self.ui.evaluateButton.connect('clicked(bool)', self.onEvaluateButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # create node to store recorded points
        self.stored_points_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "RecordedPoints")

    def cleanup(self):
        """
    Called when the application closes and the module widget is destroyed.
    """
        self.removeObservers()

    def enter(self):
        """
    Called each time the user opens this module.
    """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
    Called each time the user opens a different module.
    """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
    Called just before the scene is closed.
    """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
    Called just after the scene is closed.
    """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
    Ensure parameter node exists and observed.
    """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode):
        """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.EndModify(wasModified)

    def onInitialiseButton(self):
        self.connector = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLIGTLConnectorNode')
        self.connector.SetTypeClient("localhost", self.PORT_NUMBER)
        self.connector.Start()

        for i in range(6):
            self.curve_nodes[i] = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        default_array = np.array([
                                    [2.39574052e+00,  8.77488330e+00, - 2.01614276e+00],
                                    [-8.61602090e-01,  6.79737404e+00, - 4.99972003e-01],
                                    [-2.74751782e+00, 4.25938144e+00, 5.45960676e-02],
                                    [-4.52050827e+00,  1.18295876e+00,  3.60133685e-01],
                                    [-5.77807650e+00, - 1.37482639e+00, 6.32653479e-01],
                                    [-1.57664269e+00, - 1.09907519e-01,  1.85678359e+00],
                                    [-2.82183141e+00, - 2.54212823e+00, 1.50170447e+00],
                                    [-4.46957946e+00, - 3.31884138e+00,  2.10372522e-01],
                                    [-6.34927526e+00, - 3.20324674e+00, - 2.11814418e+00],
                                    [-2.57592881e-02, - 5.06631052e-01, 4.86522308e-01],
                                    [-1.73426941e+00, - 2.90419850e+00,  1.44023134e-03],
                                    [-4.02006172e+00, - 3.78983952e+00, - 1.71632133e+00],
                                    [-6.21477403e+00, - 3.25023383e+00, - 4.03199792e+00],
                                    [1.14153987e+00, - 1.35455478e-01, - 1.18024750e+00],
                                    [-6.69526821e-01, - 2.39284132e+00, - 1.66065004e+00],
                                    [-3.19841206e+00, - 2.43285503e+00, - 2.98162661e+00],
                                    [-5.36311232e+00, - 1.90749094e+00, - 4.38050665e+00],
                                    [1.77954752e+00, 1.10379811e+00, - 3.02524213e+00],
                                    [1.95547775e-01, - 9.25629120e-01, - 3.55482288e+00],
                                    [-2.03189626e+00, - 1.40961260e+00, - 4.26313132e+00],
                                    [-3.61769609e+00, - 9.52154491e-01, - 5.27669378e+00]]) * 10
        self.update(default_array)

    def onConnectObserverButton(self):
        # Get points
        self.points = slicer.util.getNode(self.TRACKED_OBJECT_NAME)
        self.points.AddObserver(slicer.vtkMRMLMarkupsNode.PointModifiedEvent,
                                self.processReceivedPoints)
        self.points.SetDisplayVisibility(False)

    def processReceivedPoints(self, points_mpipe, UNUSED_PARAMETER):

        # update hand coordinates - this is always done
        self.updateHand(points_mpipe)

        # record points
        if self.record is True:
            self.appendToPointRecordingArray(points_mpipe)

            # if we reach max recorded points, stop recording
            if len(self.point_recording_array) == self.NUMBER_OF_POINTS_FOR_AVG:
                self.record = False
                self.progress_bar.close()  # this closes the progress bar dialog
                self.processFinishedRecordedPoints()  # this postprocesses the results

        # trace
        if self.trace is True:
            self.updateTracing(points_mpipe)

    def updateHand(self, points_mpipe):
        if self.connector is None:
            return
        points = vtk.vtkPoints()
        points_mpipe.GetControlPointPositionsWorld(points)
        world_np = numpy_support.vtk_to_numpy(points.GetData())
        self.update(world_np)

    def update(self, world_landmarks_ori):
        """
        This function gets an ndarray of landmarks and updates the hand
        :param world_landmarks_ori:
        :return:
        """

        world_landmarks = world_landmarks_ori.copy()
        #world_landmarks = world_landmarks.astype(np.float64)
        #world_landmarks /= 1000.0

        # 2. assign loaded landmarks to anatomy
        thumb = np.asarray([world_landmarks[2], world_landmarks[3], world_landmarks[4]])
        index = np.asarray(
            [world_landmarks[5], world_landmarks[6], world_landmarks[7], world_landmarks[8]])
        middle = np.asarray(
            [world_landmarks[9], world_landmarks[10], world_landmarks[11], world_landmarks[12]])
        ring = np.asarray(
            [world_landmarks[13], world_landmarks[14], world_landmarks[15], world_landmarks[16]])
        pinky = np.asarray(
            [world_landmarks[17], world_landmarks[18], world_landmarks[19], world_landmarks[20]])
        hand = np.asarray(
            [world_landmarks[0], world_landmarks[1], world_landmarks[2], world_landmarks[5],
             world_landmarks[9], world_landmarks[13], world_landmarks[17], world_landmarks[0]])

        # 3. combine anatomies into an array, so you can loop over it
        all_sets = [thumb, index, middle, ring, pinky, hand]

        # 4. loop over all anatomies and update the corresponding curves
        empty = np.asarray([])  # we need an empty array to remove old coordinates

        for idx, curve in self.curve_nodes.items():
            # first delete old points
            slicer.util.updateMarkupsControlPointsFromArray(curve, empty)

            # then set new points
            slicer.util.updateMarkupsControlPointsFromArray(curve, all_sets[idx])

            # remove the text and change line thickness
            curve_node = slicer.mrmlScene.GetNodeByID(curve.GetID())
            disp_node = curve_node.GetDisplayNode()

            disp_node.SetTextScale(0)
            disp_node.SetCurveLineSizeMode(1)
            # disp_node.SetLineThickness(0.0001)
            disp_node.SetLineDiameter(5)

            for i in range(curve.GetNumberOfControlPoints()):
                curve.SetNthControlPointVisibility(i, False)

    def appendToPointRecordingArray(self, points_mpipe):
        points = vtk.vtkPoints()
        points_mpipe.GetControlPointPositionsWorld(points)
        world_point = numpy_support.vtk_to_numpy(points.GetData())[self.recording_finger, :]

        # append world point to the array
        self.point_recording_array.append(world_point)

        # update progressbar with current value
        self.progress_bar.value = (len(self.point_recording_array) / self.NUMBER_OF_POINTS_FOR_AVG) * 100

    def processFinishedRecordedPoints(self):
        """
        This function does outlier removal, creates new control points, updates the GUI and removes the observers.
        """

        # remove outliers
        outliers_removed = utils.outlier_removal(np.asarray(self.point_recording_array))

        # calculate mean and std
        self.recorded_points[self.current_point] = np.mean(outliers_removed, axis=0)
        self.recorded_points_stds[self.current_point] = np.std(outliers_removed, axis=0)

        dst = 0.
        if self.current_point > 0:
            dst = np.linalg.norm(self.recorded_points[self.current_point] - self.recorded_points[self.current_point - 1])

        p = self.recorded_points[self.current_point]
        std = self.recorded_points_stds[self.current_point]

        # # display new results in a message box and next to the buttons
        # slicer.util.messageBox(f"Averaged point {self.current_point+1}:\n"
        #                        f"Point:\t[{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]\n"
        #                        f"Std:\t[{std[0]:.2f}, {std[1]:.2f}, {std[2]:.2f}]\n"
        #                        f"dst:\t[{dst:.2f}]")

        # set label with all recorded points
        if self.current_point < 9:
            point_str = " " + str(self.current_point + 1)
        else:
            point_str = str(self.current_point + 1)
        self.recorded_points_labels.append(f"Point {point_str}:\t"
                                           f"[{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}],\t"
                                           f"Std: [{std[0]:.2f}, {std[1]:.2f}, {std[2]:.2f}],\t"
                                           f"dst:\t[{dst:.2f}]")
        temp_list = self.recorded_points_labels.copy()
        temp_list.reverse()
        self.point_label_string = "\n".join(temp_list)
        self.ui.pointResultLabel.setText(self.point_label_string)

        # add markups point
        self.stored_points_node.AddControlPoint(vtk.vtkVector3d(np.array(self.recorded_points[self.current_point])), f"point_{self.current_point+1}")
        self.stored_points_node.SetNthControlPointDescription(self.current_point,
                                                              f"std=[{self.recorded_points_stds[self.current_point][0]:.2f}, "
                                                              f"{self.recorded_points_stds[self.current_point][1]:.2f}, "
                                                              f"{self.recorded_points_stds[self.current_point][2]:.2f}]")
        self.current_point += 1

    def onThumbButton(self):
        try:
            self.recording_finger = Finger.thumb * 4

            self.ui.thumbButton.enabled = False
            self.ui.indexButton.enabled = True
            self.ui.middleButton.enabled = True
            self.ui.ringButton.enabled = True
            self.ui.pinkyButton.enabled = True

        except Exception as e:
            slicer.util.messageBox("Couldn't change figer.\n" + str(e))

    def onIndexButton(self):
        try:
            self.recording_finger = Finger.index * 4

            self.ui.thumbButton.enabled = True
            self.ui.indexButton.enabled = False
            self.ui.middleButton.enabled = True
            self.ui.ringButton.enabled = True
            self.ui.pinkyButton.enabled = True

        except Exception as e:
            slicer.util.messageBox("Couldn't change figer.\n" + str(e))

    def onMiddleButton(self):
        try:
            self.recording_finger = Finger.middle * 4

            self.ui.thumbButton.enabled = True
            self.ui.indexButton.enabled = True
            self.ui.middleButton.enabled = False
            self.ui.ringButton.enabled = True
            self.ui.pinkyButton.enabled = True

        except Exception as e:
            slicer.util.messageBox("Couldn't change figer.\n" + str(e))

    def onRingButton(self):
        try:
            self.recording_finger = Finger.ring * 4

            self.ui.thumbButton.enabled = True
            self.ui.indexButton.enabled = True
            self.ui.middleButton.enabled = True
            self.ui.ringButton.enabled = False
            self.ui.pinkyButton.enabled = True

        except Exception as e:
            slicer.util.messageBox("Couldn't change figer.\n" + str(e))

    def onPinkyButton(self):
        try:
            self.recording_finger = Finger.pinky * 4

            self.ui.thumbButton.enabled = True
            self.ui.indexButton.enabled = True
            self.ui.middleButton.enabled = True
            self.ui.ringButton.enabled = True
            self.ui.pinkyButton.enabled = False

        except Exception as e:
            slicer.util.messageBox("Couldn't change figer.\n" + str(e))

    def onRecordPointButton(self):
        """
        Small function that gets executed when the record button gets pressed
        """
        try:
            # clear previous values for the point and the recording array
            self.recorded_points[self.current_point] = None
            self.recorded_points_stds[self.current_point] = None
            self.point_recording_array = []

            # create progress bar
            self.progress_bar = utils.createProgressDialog(autoClose=False)
            self.progress_bar.labelText = f"Recording {self.NUMBER_OF_POINTS_FOR_AVG} points for point {self.current_point + 1}."

            # set the record value to True - then in the point process observer the recording starts
            self.record = True

        except Exception as e:
            print(e)

    def onDeletePointButton(self):
        try:

            if self.stored_points_node.GetNumberOfControlPoints() == 0:
                slicer.util.messageBox("No points to delete.")
                return

            self.stored_points_node.RemoveNthControlPoint(self.current_point - 1)  # -1 one because it gets incremented
            # in stopRecording()
            self.current_point -= 1

            self.recorded_points_labels.pop(-1)
            temp_list = self.recorded_points_labels.copy()
            temp_list.reverse()
            self.point_label_string = "\n".join(temp_list)
            self.ui.pointResultLabel.setText(self.point_label_string)

        except Exception as e:
            print(e)

    def onStartTraceButton(self):
        try:
            # create new curve
            self.trace_curves[self.trace_curve_counter] = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")

            self.trace = True
        except Exception as e:
            print(e)

    def updateTracing(self, points_mpipe):

        # Extract fingertip position
        points = vtk.vtkPoints()
        points_mpipe.GetControlPointPositionsWorld(points)
        world_point = numpy_support.vtk_to_numpy(points.GetData())[self.recording_finger, :]

        # get current curve
        curve = self.trace_curves[self.trace_curve_counter]

        # average last few points
        if curve.GetNumberOfControlPoints() >= self.TRACE_SMOOTHING:

            last_points = [world_point]
            # get last few points
            for i in range(curve.GetNumberOfControlPoints()-1, curve.GetNumberOfControlPoints() - self.TRACE_SMOOTHING - 1 ):
                last_points.append(curve.GetNthControlPointPosition(i))

            averaged_point = np.mean(np.asarray(last_points), axis=0)

            curve.AddControlPoint(averaged_point)
        else:
            curve.AddControlPoint(world_point)

        # remove the text, change line thickness and hide points
        curve_node = slicer.mrmlScene.GetNodeByID(curve.GetID())
        disp_node = curve_node.GetDisplayNode()
        disp_node.SetTextScale(0)
        disp_node.SetCurveLineSizeMode(1)
        disp_node.SetLineDiameter(3)
        for i in range(curve.GetNumberOfControlPoints()):
            curve.SetNthControlPointVisibility(i, False)

    def onStopTraceButton(self):
        self.trace = False
        self.trace_curve_counter += 1

    def onDeleteTracesButton(self):
        try:

            if self.trace_curves[0] is None:
                slicer.util.messageBox("No traces to delete")
                return

            for i in range(len(self.trace_curves)):
                if self.trace_curves[i] is None:
                    break

                slicer.mrmlScene.RemoveNode(self.trace_curves[i])
                self.trace_curves[i] = None

            self.trace_curve_counter = 0

        except Exception as e:
            print(e)

    def onPrintResultsButton(self):
        try:

            split_results = self.point_label_string.split('\n')

            print("All  points:")
            print("x,y,z")
            for i in range(len(split_results)):
                coors = split_results[i].split('\t')
                p = coors[1].split('[')[1].split(']')[0].split(',')
                print(f"{p[0]},{p[1][1:]},{p[2][1:]}")

        except Exception as e:
            slicer.util.messageBox("Couldn't print results.\n" + str(e))

    def onEvaluateButton(self):

        try:
            ground_truth_list = slicer.util.getNode(self.ui.groundTruthText.toPlainText())
            if ground_truth_list is None:
                raise Exception("Ground truth list not found.")
            ground_truth_list = np.array([ground_truth_list.GetNthControlPointPositionVector(i) for i in range(ground_truth_list.GetNumberOfControlPoints())])

            handnav_list = slicer.util.getNode(self.ui.handNavText.toPlainText())
            if handnav_list is None:
                raise Exception("HandNav list not found.")
            handnav_list = np.array([handnav_list.GetNthControlPointPositionVector(i) for i in range(handnav_list.GetNumberOfControlPoints())])

            indices_lef_out = [int(i) for i in self.ui.idxLeftOutText.toPlainText().split(',') if i != '']
            indices_reg_all = [int(i) for i in self.ui.idxText.toPlainText().split(',')]
            indices_reg = []
            indices_eval = []

            assert len(handnav_list) == len(ground_truth_list), "Both lists must have the same length."

            for i in range(len(handnav_list)):
                if i not in indices_reg_all and i not in indices_lef_out:
                    indices_eval.append(i)

                if i in indices_reg_all and i not in indices_lef_out:
                    indices_reg.append(i)

            assert len(indices_reg) > 0, "No reg indices selected."
            assert len(indices_eval) > 0, "No eval indices selected."

            registration_mse = rms_error(ground_truth_list[indices_reg, :], handnav_list[indices_reg, :])
            rest_mse = rms_error(ground_truth_list[indices_eval, :], handnav_list[indices_eval, :])

            # distances from HandNav points to NousNav points - we calculate those only for the 'unregistered' points
            registration_distances = np.sqrt(
                np.sum((ground_truth_list[indices_reg, :] - handnav_list[indices_reg, :]) ** 2, axis=1))
            rest_distances = np.sqrt(
                np.sum((ground_truth_list[indices_eval, :] - handnav_list[indices_eval, :]) ** 2, axis=1))

            # mean, std, median, min, max
            mean_distance_registration = np.mean(registration_distances)
            std_distance_registration = np.std(registration_distances)
            median_distance_registration = np.median(registration_distances)
            min_distance_registration = np.min(registration_distances)
            max_distance_registration = np.max(registration_distances)

            mean_distance_rest = np.mean(rest_distances)
            std_distance_rest = np.std(rest_distances)
            median_distance_rest = np.median(rest_distances)
            min_distance_rest = np.min(rest_distances)
            max_distance_rest = np.max(rest_distances)

            np.set_printoptions(precision=2)

            print(f"Registration mse: {registration_mse:.2f}")
            print(f"Rest mse: {rest_mse:.2f}\n")

            print(f"Registration distances: {registration_distances}")
            print(f"Mean registration distance: {mean_distance_registration:.2f}")
            print(f"Std registration distance: {std_distance_registration:.2f}")
            print(f"Median registration distance: {median_distance_registration:.2f}")
            print(f"Min registration distance: {min_distance_registration:.2f}")
            print(f"Max registration distance: {max_distance_registration:.2f}\n")

            print(f"Rest distances: {rest_distances}")
            print(f"Mean rest distance: {mean_distance_rest:.2f}")
            print(f"Std rest distance: {std_distance_rest:.2f}")
            print(f"Median rest distance: {median_distance_rest:.2f}")
            print(f"Min rest distance: {min_distance_rest:.2f}")
            print(f"Max rest distance: {max_distance_rest:.2f}")

        # TODO add field to exclude landmarks
        except Exception as e:
            slicer.util.messageBox("Couldn't evaluate.\n" + str(e))


#
# X-NavLogic
#

class HandNavLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def __init__(self):
        """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
        ScriptedLoadableModuleLogic.__init__(self)


#
# X-NavTest
#
#
class HandNavTest(ScriptedLoadableModuleTest):
    """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
    """
        self.setUp()
        self.test_HandNav1()

    def test_HandNav1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

        self.delayDisplay("Starting the test")

        logic = HandNavLogic()

        pass

        self.delayDisplay('Test passed')


class Finger(enum.IntEnum):
    thumb = 1
    index = 2
    middle = 3
    ring = 4
    pinky = 5
    centerOfTheUniverse = 6


def rms_error(pred, actual):

    assert isinstance(pred, np.ndarray) and isinstance(actual, np.ndarray), "Prediction and actual must be numpy arrays"
    assert pred.shape == actual.shape, "Prediction and actual must be the same shape."

    return np.sqrt(np.mean(np.sum((pred - actual) ** 2)))

