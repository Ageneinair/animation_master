#!/usr/bin/env python
import os
import sys
import cv2
import numpy as np
import random
import h5py
from argparse import ArgumentParser

# Import Mask RCNN
ROOT_DIR_MASK = os.path.dirname(os.path.realpath(__file__))
# ROOT_DIR_MASK = os.path.abspath("../")
ROOT_DIR_MASK += "/Mask_RCNN"
sys.path.append(ROOT_DIR_MASK)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
import mrcnn.model as modellib

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR_MASK, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR_MASK, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR_MASK, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# Mask RCNN
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class RoisDetector(object):
    def __init__(self):
        self.config = InferenceConfig()
        # self.config.display()
        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=self.config)
        # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)

        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        self.class_names = ['BG', 'person']

    def get_rois(self, origin_image):
        # Run detection
        results = self.model.detect([origin_image], verbose=0)
        r = results[0]
        rois = []
        for i in range(len(r['class_ids'])):
            y1, x1, y2, x2 = r['rois'][i]
            rois.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        return rois


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image", default='sup-mat/tai-chi.jpg', help="Path to input image")
    opt = parser.parse_args()
    image = cv2.imread(opt.image)

    rd = RoisDetector()
    rois = rd.get_rois(image)
    
    for roi in rois:
        (x1, y1, x2, y2) = roi
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)
    cv2.imwrite('find_rois.png',image)
    cv2.imshow('ROIS', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


