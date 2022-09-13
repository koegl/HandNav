import numpy as np
import qt
from slicer import app


def outlier_removal(nd_array_of_points, factor=2.0, method='mean'):
    """
    This function calculates removes outliers from a numpy array of points based on the mean and std.
    Code for the median method is taken from: https://stackoverflow.com/a/16562028/14293274
    :param nd_array_of_points: Nx3 nd_array of points
    :param factor: factor to multiply the std by
    :param method: method to use for outlier removal (mean or median)
    :return: the reduced points
    """

    assert nd_array_of_points.shape[1] == 3, "nd_array_of_points must be Nx3"
    assert method in ['mean', 'median'], "method must be 'mean' or 'median'"

    # create a middle point
    if method == 'mean':
        middle_point = np.mean(nd_array_of_points, axis=0)
    else:
        middle_point = np.median(nd_array_of_points, axis=0)

    # calculate distance between each point and the middle point
    distances = np.sqrt(np.sum((nd_array_of_points - middle_point) ** 2, axis=1))

    # calculate the threshold
    if method == 'mean':
        standard_deviation = np.std(distances)
        threshold = factor * standard_deviation
    else:  # median absolute distance to the median
        median = np.median(distances)
        threshold = distances / median

    # choose all points where their distance to the mean point is less than the threshold
    if method == 'mean':
        reduced_array = [nd_array_of_points[i, :] for i in range(len(distances)) if distances[i] < threshold]
        reduced_array = np.array(reduced_array)
    else:
        reduced_array = nd_array_of_points[threshold < factor]

    return reduced_array


def lookupTopLevelWidget(objectName):
    """
    Loop over all top level widget associated with 'slicer.app' and
    return the one matching 'objectName'
    :raises RuntimeError: if no top-level widget is found by that name

    # taken from https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/util.py#L1136
    """

    for w in app.topLevelWidgets():
        if hasattr(w, 'objectName'):
            if w.objectName == objectName:
                return w
    # not found
    raise RuntimeError("Failed to obtain reference to '%s'" % objectName)


def mainWindow():
    """
    Get main window widget (qSlicerMainWindow object)
    :return: main window widget, or ``None`` if there is no main window

    # taken from https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/util.py#L1136
    """

    try:
        mw = lookupTopLevelWidget('qSlicerMainWindow')
    except RuntimeError:
        # main window not found, return None
        # Note: we do not raise an exception so that this function can be conveniently used
        # in expressions such as `parent if parent else mainWindow()`
        mw = None
    return mw


def createProgressDialog(parent=None, value=0, maximum=100, labelText="", windowTitle="Processing...", **kwargs):
    """
    Display a modal QProgressDialog.
    Go to `QProgressDialog documentation <https://doc.qt.io/qt-5/qprogressdialog.html>`_ to
    learn about the available keyword arguments.
    Examples::
        # Prevent progress dialog from automatically closing
        progressbar = createProgressDialog(autoClose=False)
        # Update progress value
        progressbar.value = 50
        # Update label text
        progressbar.labelText = "processing XYZ"

    # taken from https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/util.py#L1136
    """

    progressIndicator = qt.QProgressDialog(parent if parent else mainWindow())
    progressIndicator.minimumDuration = 0
    progressIndicator.maximum = maximum
    progressIndicator.value = value
    progressIndicator.windowTitle = windowTitle
    progressIndicator.labelText = labelText

    for key, value in kwargs.items():
        if hasattr(progressIndicator, key):
            setattr(progressIndicator, key, value)

    return progressIndicator
