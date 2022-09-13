import cv2
import mediapipe as mp
import numpy as np
import scipy.signal

from utils import key_stroke_logic, Hand, User
from personal_configs import focal_length, principal_point, distortion, camera_resolution, user


"""=====================================================================================================================
                                                    PARAMETERS
====================================================================================================================="""
adjust_depth = True
adjust_length = True
assume_fingers_are_straight = True
number_of_straight_fingers = 4
low_pass = True
current_user = User(user)


"""=====================================================================================================================
                                                    INITIALIZATION
====================================================================================================================="""
camera_matrix = np.array(
                         [[focal_length[0], 0, principal_point[0]],
                          [0, focal_length[1], principal_point[1]],
                          [0, 0, 1]],
                         dtype="double"
                         )
hand = Hand(camera_matrix, distortion, camera_resolution)
cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_ANY,
                       params=[cv2.CAP_PROP_FRAME_WIDTH, camera_resolution[0],
                               cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution[1]])

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


"""=====================================================================================================================
                                                    TRACKING
====================================================================================================================="""
with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():

        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:

            hand.extract_mpipe_landmarks(results)

            if assume_fingers_are_straight:
                hand.straighten_fingers(number_of_straight_fingers)

            hand.solve_pnp()

            hand.calculate_world_coordinates()

            if adjust_depth:
                hand.push_coordinates_away(current_user.half_hand_depth_mm)
            if adjust_length:
                hand.extend_fingers(current_user.tip_extension_length_mm)

            hand.strip_homogeneous_from_world_points()

            if low_pass:
                hand.low_pass_filter()

            hand.send_coordinates()

            # draw all the points and lines on the image
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

        # get user inputs to stop or print a location
        key_stroke = cv2.waitKey(5)

        quit_val, adjust_depth, adjust_length, current_user, low_pass = key_stroke_logic(key_stroke, adjust_depth, adjust_length, current_user, low_pass)

        if quit_val is True:
            break

cap.release()
