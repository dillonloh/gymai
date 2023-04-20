import matplotlib.pyplot as plt
import matplotlib.animation as animation

from modules.keypoints_processor import *
from icecream import ic

import cv2


class DepthPlot:
    """
    class for plotting depth graphs/animations
    """

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
        """
        Add important keypoints based on what movement is being done.
        """
        if self.movement == 'squat':
            self._add_squat_keypoints(keypoints_with_scores)


    def _add_squat_keypoints(self, keypoints_with_scores):

        for name, id in self.ids.items():
            self.keypoints[name].append(preprocess_keypoint(keypoints_with_scores[0][0][id][:2]))


    def plot_depth(self, filename='test.mp4'):
        """
        Plot a graph of relative depth vs frame
        Relative depth is defined based on movement type
        """

        right_array = np.array(self.keypoints['right_hip'])[:, 1] - np.array(self.keypoints['right_knee'])[:, 1]
        left_array = np.array(self.keypoints['left_hip'])[:, 1] - np.array(self.keypoints['left_knee'])[:, 1]

        plt.plot(right_array, color='green', label='Right Leg')
        plt.plot(left_array, color='blue', label='Left Leg')

        plt.hlines(y=0, xmin=0, xmax=len(self.keypoints['right_hip']), label='Parallel Depth', color='gray', linestyle='dashed')
        plt.legend()

        plt.ylabel("Relative Depth")
        plt.xlabel("Video Frames")

        plt.title('Relative Depth of Hip against Knee', fontsize=16)
        plt.savefig(filename)
        plt.show()


    def plot_animation(self, filename='test.mp4'):
        """
        Plot an animated graph of relative depth vs frame
        Relative depth is defined based on movement type
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)

        left_depth = np.array(self.keypoints['left_hip'])[:, 1] - np.array(self.keypoints['left_knee'])[:, 1]
        right_depth =  np.array(self.keypoints['right_hip'])[:, 1] - np.array(self.keypoints['right_knee'])[:, 1]

        line1, = ax.plot(left_depth, alpha=0.7, color='green', label='Right Leg')
        line2, = ax.plot(right_depth, alpha=0.5, color='blue', label='Left Leg')

        ax.hlines(y=0, xmin=0, xmax=len(self.keypoints['right_hip']), label='Parallel Depth', color='gray', linestyle='dashed')

        ax.set_ylabel("Relative Depth")
        ax.set_xlabel("Video Frames")
        ax.set_title('Relative Depth of Hip against Knee', fontsize=16)

        ax.legend()
        x = np.array(list(range(left_depth.shape[0])))

        def updateline(num, x, left_depth, right_depth, line1, line2):
            # print(data[0][..., :num])
            # ic(left_depth[:num].shape)
            # ic(np.array(list(range(num))).shape)
            line1.set_data(x[:num], left_depth[:num])
            line2.set_data(x[:num], right_depth[:num])
            
            return line1, line2
        
        line_animation = animation.FuncAnimation(
        fig, updateline, interval=10, fargs=(x, left_depth, right_depth, line1, line2), blit=True, repeat=True)

        line_animation.save(filename)
        plt.show()