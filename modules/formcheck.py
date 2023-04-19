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

# local modules
from modules.keypoints_processor import *

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

	return (squat_keypoints['left_hip'][1] < squat_keypoints['left_knee'][1] and squat_keypoints['right_hip'][1] < squat_keypoints['right_knee'][1])


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
	
	return (bench_keypoints['left_elbow'][1] < bench_keypoints['left_shoulder'][1] and bench_keypoints['right_elbow'][1] < bench_keypoints['right_shoulder'][1])



