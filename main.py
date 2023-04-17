import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import numpy as np
import cv2
import math
import sys
import time

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# debug
from icecream import ic

# import local modules
from modules.visualisation import *
from modules.formcheck import *
from modules.inference import *

# Initialize the TFLite interpreter
interpreter = tf.lite.Interpreter(model_path="models/model.tflite")
interpreter.allocate_tensors()
input_size = 192


# Load the input image.
video_path = 'videos/squat_dillon.mp4'
CURRENT_MOVEMENT = 'squat' # 'squat' or 'bench'

# Read the video from specified path
cam = cv2.VideoCapture(video_path)

frames = []
n_frames = 0
while True:
    ret, frame = cam.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        n_frames += 1
    else:
       break
print(n_frames, 'frames processed')

# convert frames array to tensor
image = tf.convert_to_tensor(np.array(frames))
ic(image.shape)

num_frames, image_height, image_width, _ = image.shape
crop_region = init_crop_region(image_height, image_width)

# Run model inference.
depth_flag = False
output_images = []
for frame_idx in range(num_frames):

    # status bar
    print("{:.1f}%".format((100/(num_frames-1)*frame_idx)), end='\r')

    keypoints_with_scores = run_inference(
        movenet, image[frame_idx, :, :, :], crop_region,
        crop_size=[input_size, input_size], interpreter=interpreter)

    crop_region = determine_crop_region(
      keypoints_with_scores, image_height, image_width)

    if CURRENT_MOVEMENT == 'squat':
        depth_flag = True if check_squat_depth(keypoints_with_scores) else depth_flag

    elif CURRENT_MOVEMENT == 'bench':
        depth_flag = True if check_bench_depth(keypoints_with_scores) else depth_flag

    output_images.append(draw_prediction_on_image(
        image[frame_idx, :, :, :].numpy().astype(np.int32),
        keypoints_with_scores, crop_region=None,
        close_figure=True, output_image_height=300, depth_flag=depth_flag))

print('\n')
if depth_flag == True:
    print("Depth good!")

else: 
    print("Insufficient Depth")

output = np.stack(output_images, axis=0)
to_video(output, fps=30, name='inference/'+video_path.split('/')[-1])