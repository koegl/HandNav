import numpy as np
from numpy import sin, cos
import scipy
import cv2
import enum
import pyigtl


class Finger(enum.IntEnum):
    thumb = 1
    index = 2
    middle = 3
    ring = 4
    pinky = 5
    centerOfTheUniverse = 6


class Users(enum.IntEnum):
    user_1 = 1


class User:
    def __init__(self, user):

        self.user = user
        self.hand_depth_mm = 0.
        self.half_hand_depth_mm = 0.
        self.tip_extension_length_mm = 0.
        self.set_params(user)

    def set_params(self, user):
        if user == Users.user_1:
            self.half_hand_depth_mm = 5.
            self.tip_extension_length_mm = 11.


def extract_mpipe_landmarks(mpipe_landmarks, dim=3):
    """
    Extracts mpipe landmarks into a Nx3 numpy array. Example usage:
    nd_landmarks = extract_mpipe_landmarks(results.multi_hand_landmarks)
    :param mpipe_landmarks: the landmarks from hands.process(image): results.multi_hand_landmarks or
    results.multi_hand_world_landmarks
    :param dim: the dimension of the landmarks.
    :return: Nx3 numpy array with landmarks
    """

    landmarks = mpipe_landmarks[0].landmark

    if dim == 3:
        np_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
    elif dim == 2:
        np_landmarks = np.array([(lm.x, lm.y) for lm in landmarks])
    else:
        raise ValueError("Dimension must be 2 or 3.")

    return np_landmarks


def key_stroke_logic(key_stroke, adjust_depth, adjust_length, user, low_pass):

    quit_val = False

    if key_stroke & 0xFF == 27:
        quit_val = True

    elif key_stroke == ord('f'):
        user.set_params(Users.user_1)
        print("Switched to user: user_1")

    elif key_stroke == ord('l'):
        low_pass = True
        print("Low pass filter enabled")
    elif key_stroke == ord('h'):
        low_pass = False
        print("Low pass filter disabled")

    elif key_stroke == ord('n'):
        adjust_depth = True
        print("Starting depth compensation")
    elif key_stroke == ord('m'):
        adjust_depth = False
        print("Stopping depth compensation")

    elif key_stroke == ord('z'):
        adjust_length = True
        print("Starting length compensation")
    elif key_stroke == ord('x'):
        adjust_length = False
        print("Stopping length compensation")

    return quit_val, adjust_depth, adjust_length, user, low_pass


class LowPassFilter:
    def __init__(self, fs=10, low_cut=0.05, order=2, data_shape=(21, 3)):

        nyq = 0.5 * fs
        low = low_cut / nyq

        self.b, self.a = scipy.signal.butter(order, low, 'lowpass', analog=False)

        self.f_state = np.zeros((order, data_shape[0], data_shape[1]))

    def execute(self, signal):
        y, self.f_state = scipy.signal.lfilter(self.b, self.a, signal, axis=0, zi=self.f_state)

        return y


class Hand:
    """
    Class to encapsulate all functionality required to transform image and model coordinates into 3D world coordinates
    that can be sent to Slicer - this includes solving PnP, finger extensions, depth extension, low-pass filtering and
    communication with Slicer.
    """

    def __init__(self, camera_matrix, distortion, camera_resolution, pnp_flag=cv2.SOLVEPNP_SQPNP, unit_conversion=1000,
                 fs=10, low_cut=1, order=2, data_shape=(21, 3), filter_packet_length=2,
                 port=18945,
                 ):

        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.camera_resolution = camera_resolution
        self.pnp_flag = pnp_flag
        self.unit_conversion = unit_conversion
        self.filter_packet_length = filter_packet_length
        self.half_hand_depth_mm = 5
        self.tip_extension_length_mm = 7

        # points
        self.image_points = None
        self.model_points = None
        self.world_points = None

        # transformations
        self.rotation_vector = None
        self.translation_vector = None
        self.transformation = np.eye(4)

        # filter
        self.low_pass_filter_instance = LowPassFilter(fs=fs, low_cut=low_cut, order=order, data_shape=data_shape)
        self.signal = np.zeros((self.filter_packet_length, 21, 3))
        self.signal_counter = 0

        # communication
        self.server = pyigtl.OpenIGTLinkServer(port)

    def extract_mpipe_landmarks(self, results):
        self.image_points = extract_mpipe_landmarks(results.multi_hand_landmarks, dim=2) * self.camera_resolution
        self.model_points = extract_mpipe_landmarks(results.multi_hand_world_landmarks, dim=3) * self.unit_conversion

    def straighten_fingers(self, num_fingers=4):
        if num_fingers < 4:
            num_fingers = 4
        for idx in range(num_fingers):
            tip_idx = (idx+2)*4
            correct_direction = self.model_points[tip_idx-2] - self.model_points[tip_idx-3]
            correct_direction_normalized = correct_direction / np.linalg.norm(correct_direction)
            length76 = np.linalg.norm(self.model_points[tip_idx-1] - self.model_points[tip_idx-2])
            length87 = np.linalg.norm(self.model_points[tip_idx] - self.model_points[tip_idx-1])
            corrected7 = self.model_points[tip_idx-2] + length76 * correct_direction_normalized
            self.model_points[tip_idx-1] = corrected7
            corrected8 = corrected7 + length87 * correct_direction_normalized
            self.model_points[tip_idx] = corrected8

    def solve_pnp(self):
        assert self.image_points is not None and self.model_points is not None,\
               "Points must be extracted before solving PnP"

        _, self.rotation_vector, self.translation_vector = cv2.solvePnP(self.model_points,
                                                                        self.image_points,
                                                                        self.camera_matrix,
                                                                        self.distortion,
                                                                        flags=self.pnp_flag)

        self.transformation[0:3, 3] = -self.translation_vector.squeeze()

    def calculate_world_coordinates(self):
        model_points_hom = np.concatenate((self.model_points, np.ones((21, 1))), axis=1)
        self.world_points = model_points_hom.dot(np.linalg.inv(self.transformation).T)

        # check if any z-coordinates are flipped - if yes, redo the PnP with flipped translation
        if np.any(self.world_points[:, 2] < 0):
            self.transformation[0:3, 3] = self.translation_vector.squeeze()
            self.world_points = model_points_hom.dot(np.linalg.inv(self.transformation).T)

    def push_coordinates_away(self, push_distance):
        # Push points away from camera by half-hand depth:
        normalized_translation_direction = self.translation_vector / np.linalg.norm(self.translation_vector)
        transformation_dc = np.eye(4)
        transformation_dc[0:3, 3] = -normalized_translation_direction.squeeze() * push_distance
        self.world_points = self.world_points.dot(np.linalg.inv(transformation_dc).T)

    def extend_fingers(self, extend_distance):
        for i in range(5):
            current_tip_index = (i + 1) * 4
            extension_direction = self.world_points[current_tip_index, 0:3] - self.world_points[current_tip_index - 1, 0:3]
            extension_direction_normalized = extension_direction / np.linalg.norm(extension_direction)
            self.world_points[current_tip_index, 0:3] += extension_direction_normalized.squeeze() * extend_distance

    def strip_homogeneous_from_world_points(self):
        self.world_points = self.world_points[:, 0:3]

    def low_pass_filter(self):
        # add new world_points to signal
        self.signal = self.signal[1:, :, :]  # remove first hand-array
        self.signal = np.append(self.signal, np.expand_dims(self.world_points, axis=0), axis=0)  # append new hand array

        if self.signal_counter >= self.filter_packet_length - 1:
            signal_filtered = self.low_pass_filter_instance.execute(self.signal)
            self.world_points = signal_filtered[self.filter_packet_length//2, :, :]

        self.signal_counter += 1

    def send_coordinates(self):
        OIGTLMessage = pyigtl.PointMessage(self.world_points)
        self.server.send_message(OIGTLMessage)
