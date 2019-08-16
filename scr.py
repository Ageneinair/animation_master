import cv2
import numpy as np

cv2.imwrite('taichisource.png', cv2.resize(src=cv2.imread('testtwo.jpg'), dst=None, dsize=(256,256)))
cv2.imwrite('taichisourceresized.png', cv2.resize(src=cv2.imread('testtwo.jpg'), dst=None, dsize=(128,64)))
