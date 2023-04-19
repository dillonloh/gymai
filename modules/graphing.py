import matplotlib.pyplot as plt
from modules.keypoints_processor import *
from icecream import ic

class DepthPlot:

    def __init__(self, movement):

        self.movement = movement
        if self.movement == 'squat':
            self.keypoints = {
                'left_hip' : [],
                'left_knee': [],
                'left_ankle': [],
                'right_hip': [],
                'right_knee': [],
                'right_ankle': [],
                }
            
            self.ids = {
                    'left_hip' : 11,
                    'left_knee': 13,
                    'left_ankle': 15,
                    'right_hip': 12,
                    'right_knee': 14,
                    'right_ankle': 16,
                }


    def add_keypoints(self, keypoints_with_scores):

        if self.movement == 'squat':
            self._add_squat_keypoints(keypoints_with_scores)


    def _add_squat_keypoints(self, keypoints_with_scores):

        for name, id in self.ids.items():
            self.keypoints[name].append(preprocess_keypoint(keypoints_with_scores[0][0][id][:2]))


    def plot_depth(self, filename='test.mp4'):

        right_array = np.array(self.keypoints['right_hip'])[:, 1] - np.array(self.keypoints['right_knee'])[:, 1]
        left_array = np.array(self.keypoints['left_hip'])[:, 1] - np.array(self.keypoints['left_knee'])[:, 1]

        plt.plot(right_array, color='green', label='Right Leg')
        plt.plot(left_array, color='blue', label='Left Leg')

        plt.hlines(y=0, xmin=0, xmax=len(self.keypoints['right_hip']), label='Parallel Depth', color='gray', linestyle='dashed')
        plt.legend()

        plt.title('Relative Depth of Hip against Knee', fontsize=16)
        plt.savefig(filename)
        plt.show()
