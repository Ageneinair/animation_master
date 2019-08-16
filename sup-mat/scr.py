import numpy as np
import cv2
import imageio

image = cv2.imread('driving.png')[:,:,::-1]
frames_list = []
for i in range(image.shape[1] // 128):
    frames_list.append(image[:, i*128:(i+1)*128])

imageio.mimsave('driving_black_horse.gif', frames_list)
