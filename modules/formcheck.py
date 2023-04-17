import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import math

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# debug
from icecream import ic

def preprocess_keypoint(keypoint):
	"""
	convert model output coordinates from yx (towards bottom right) coordinate system to xy (towards top left coordinate system)
	"""

	keypoint = np.flipud(keypoint) # flip yx to xy
	keypoint[1] = 1 - keypoint[1] # flip y direction

	return keypoint


def unit_vector(vector):
    """ 
	Returns the unit vector of the vector.  
	"""
    return vector / np.linalg.norm(vector)


def calculate_angle(point1, point2, joint):
	"""
	function for calculating angle between 2 keypoints about one joint (joint)
	by default returns the angle in range of (0, 180) degrees (i.e. the smaller one)
	"""

	jp1 = point1 - joint
	jp2 = point2 - joint

	jp1_unit = unit_vector(jp1)
	jp2_unit = unit_vector(jp2)

	angle = np.arccos(np.dot(jp1_unit, jp2_unit)) * (180/math.pi)
        
	return angle if angle <= 180 else (360 - angle)


def check_squat_depth(keypoints_with_scores):
	"""
	Function for checking the depth of a squat if it hits parallel
	"""
	squat_ids = {
				'left_hip' : 11,
				'left_knee': 13,
				'left_ankle': 15,
				'right_hip': 12,
				'right_knee': 14,
				'right_ankle': 16,
	}

	squat_keypoints = {}

	for name, id in squat_ids.items():
		squat_keypoints[name] = preprocess_keypoint(keypoints_with_scores[0][0][id][:2])

	left_squat_angle = calculate_angle(squat_keypoints['left_hip'], squat_keypoints['left_ankle'], squat_keypoints['left_knee'])
	right_squat_angle = calculate_angle(squat_keypoints['right_hip'], squat_keypoints['right_ankle'], squat_keypoints['right_knee'])

	return (left_squat_angle <= 90 or right_squat_angle <= 90)


def check_bench_depth(keypoints_with_scores):
	"""
	function for checking the depth of a bench if elbow goes below shoulder
	"""
	bench_ids = {
		    'left_shoulder': 5,
			'right_shoulder': 6,
			'left_elbow': 7,
			'right_elbow': 8,
			'left_wrist': 9,
			'right_wrist': 10,
	}

	bench_keypoints = {}

	for name, id in bench_ids.items():
		bench_keypoints[name] = preprocess_keypoint(keypoints_with_scores[0][0][id][:2])
	
	return (bench_keypoints['left_elbow'][1] < bench_keypoints['left_shoulder'][1] or bench_keypoints['right_elbow'][1] < bench_keypoints['right_shoulder'][1])



