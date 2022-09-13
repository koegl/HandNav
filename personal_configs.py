import numpy as np
from utils import Users

folders = []

print("user_1")

focal_length = np.load("camera_calibrations/calibrations_20220618/user_1_focal_length.npy")
principal_point = np.load("camera_calibrations/calibrations_20220618/user_1_principal_point.npy")
distortion = np.load("camera_calibrations/calibrations_20220618/user_1_distortion.npy")
camera_resolution = np.array([1920, 1080])

user = Users.user_1
