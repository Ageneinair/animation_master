import cv2
import numpy as np

image = cv2.imread('transformed.png')
gt = cv2.imread('gt.png')

aed = np.mean(np.sqrt((image[:, :, 0]-gt[:, :, 0])**2 + (image[:, :, 1]-gt[:, :, 1])**2 + (image[:, :, 2]-gt[:, :, 2])**2))
print("AED: " + str(aed))

l1 = np.mean(np.abs(image - gt))
print("L1: " + str(l1))
