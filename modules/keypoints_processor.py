import numpy as np
from icecream import ic

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


def preprocess_keypoint(keypoint):
	"""
	convert model output coordinates from yx (towards bottom right) coordinate system to xy (towards top left coordinate system)
	"""

	processed_keypoint = np.copy(keypoint)
	processed_keypoint = np.flipud(processed_keypoint) # flip yx to xy
	processed_keypoint[1] = 1 - processed_keypoint[1] # flip y direction

	return processed_keypoint

