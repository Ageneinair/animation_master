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

# Import all Monkey-Net resources
import matplotlib
matplotlib.use('Agg')
import torch
from transfer import transfer_one
import yaml
from logger import Logger
from modules.generator import MotionTransferGenerator
from modules.keypoint_detector import KPDetector
import imageio
from sync_batchnorm import DataParallelWithCallback
from frames_dataset import read_video
from augmentation import VideoToTensor





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
    parser.add_argument("--driving_video", required = True, help = 'The driving video')
    parser.add_argument("--config", required = True, help = "The configuration")
    parser.add_argument("--checkpoint", required = True, help = "The checkpoint")
    parser.add_argument("--image_shape", default = (128, 128), type = lambda x: tuple([int(a) for a in x.split(',')]), help = 'Image shape')
    parser.add_argument("--out_file", default = 'demo', help = "Name of output file without format")
    parser.add_argument("--cpu", dest = "cpu", action = "store_true", help = "Use cpu")

    opt = parser.parse_args()
    image = cv2.imread(opt.image)

    rd = RoisDetector()
    rois = rd.get_rois(image)
    img = np.copy(image)
    

    with open(opt.config) as f:
            config = yaml.load(f)
            blocks_discriminator = config['model_params']['discriminator_params']['num_blocks']
            assert len(config['train_params']['loss_weights']['reconstruction']) == blocks_discriminator + 1

    generator = MotionTransferGenerator(**config['model_params']['generator_params'], 
                                            **config['model_params']['common_params'])

    if not opt.cpu:
        generator = generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'], 
                                 **config['model_params']['common_params'])
    if not opt.cpu:
        kp_detector = kp_detector.cuda()

    Logger.load_cpk(opt.checkpoint, generator = generator, kp_detector = kp_detector)

    generator = DataParallelWithCallback(generator)
    kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()



    for i, roi in enumerate(rois):
        (x1, y1, x2, y2) = roi
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)
        #cv2.imwrite('object' + str(i) + '.png', img[y1:y2, x1:x2])

        """ HERE RUN MONKEY-NET ON THE SEPARATE OBJECT """
        inp_img = np.zeros(shape = opt.image_shape)
        inp_img = cv2.resize(src = img[y1:y2, x1:x2], dst = inp_img, dsize = inp_img.shape)
        cv2.imwrite('object' + str(i) + '.png', inp_img)

        with torch.no_grad():
            driving_video = VideoToTensor()(read_video(opt.driving_video, opt.image_shape + (3, )))['video']
            source_image = VideoToTensor()(read_video('object' + str(i) + '.png', opt.image_shape + (3, )))['video'][:, :1]
            #print("source image shape " + str(source_image.shape))
            #print("inp_img shape " + str(inp_img.shape))

            driving_video = torch.from_numpy(driving_video).unsqueeze(0)
            source_image = torch.from_numpy(source_image).unsqueeze(0)

            out = transfer_one(generator, kp_detector, source_image, driving_video, config['transfer_params'])

            out_video_batch = out['video_prediction'].data.cpu().numpy()
            out_video_batch = np.transpose(out_video_batch, [0, 2, 3, 4, 1])[0]
            imageio.mimsave(opt.out_file + str(i) + '.gif', (255 * out_video_batch).astype(np.uint8))


    print("Created " + str(len(rois)) + " new gifs.")
    cv2.imwrite('find_rois.png',image)
    #cv2.imshow('ROIS', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


