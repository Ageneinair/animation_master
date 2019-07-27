import torch
from torch import nn
import torch.nn.functional as F

from modules.util import Encoder, Decoder, ResBlock3D
from modules.dense_motion_module import DenseMotionModule, IdentityDeformation
from modules.movement_embedding import MovementEmbeddingModule

from matplotlib import pyplot as plt
import numpy as np
import sys


class MotionTransferGenerator(nn.Module):
    """
    Motion transfer generator. That Given a keypoints and an appearance trying to reconstruct the target frame.
    Produce 2 versions of target frame, one warped with predicted optical flow and other refined.
    """

    def __init__(self, num_channels, num_kp, kp_variance, block_expansion, max_features, num_blocks, num_refinement_blocks,
                 dense_motion_params=None, kp_embedding_params=None, interpolation_mode='nearest'):
        super(MotionTransferGenerator, self).__init__()

        print("init Motion Transfer Generator")

        self.appearance_encoder = Encoder(block_expansion, in_features=num_channels, max_features=max_features,
                                          num_blocks=num_blocks)

        if kp_embedding_params is not None:
            self.kp_embedding_module = MovementEmbeddingModule(num_kp=num_kp, kp_variance=kp_variance,
                                                               num_channels=num_channels, **kp_embedding_params)
            embedding_features = self.kp_embedding_module.out_channels
        else:
            self.kp_embedding_module = None
            embedding_features = 0

        if dense_motion_params is not None:
            self.dense_motion_module = DenseMotionModule(num_kp=num_kp, kp_variance=kp_variance,
                                                         num_channels=num_channels,
                                                         **dense_motion_params)
        else:
            self.dense_motion_module = IdentityDeformation()

        self.video_decoder = Decoder(block_expansion=block_expansion, in_features=num_channels,
                                     out_features=num_channels, max_features=max_features, num_blocks=num_blocks,
                                     additional_features_for_block=embedding_features,
                                     use_last_conv=False)

        self.refinement_module = torch.nn.Sequential()
        in_features = block_expansion + num_channels + embedding_features
        for i in range(num_refinement_blocks):
            self.refinement_module.add_module('r' + str(i),
                                              ResBlock3D(in_features, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        self.refinement_module.add_module('conv-last', nn.Conv3d(in_features, num_channels, kernel_size=1, padding=0))
        self.interpolation_mode = interpolation_mode

    def deform_input(self, inp, deformations_absolute):
        bs, d, h_old, w_old, _ = deformations_absolute.shape
        #print(str(deformations_absolute.shape))

        _, _, _, h, w = inp.shape

        #plt.imsave("inp.png", inp.data[0, 0, 0].cpu().numpy())

        deformations_absolute = deformations_absolute.permute(0, 4, 1, 2, 3)
        deformation = F.interpolate(deformations_absolute, size=(d, h, w), mode=self.interpolation_mode)
        #print(str(deformation.shape))
        #plt.imsave("deformation.png", deformation[0, 0, 0].cpu().numpy())
        deformation = deformation.permute(0, 2, 3, 4, 1)
        
        print(str(deformation.shape))
        #deform = deformation[0, 0].cpu().numpy()
        #plt.imsave("deformation.png", np.sqrt(deform[:,:,0] ** 2 + deform[:,:,1] ** 2 + deform[:,:,2] ** 2))

        deformed_inp = F.grid_sample(inp, deformation)
        #print(str(deformed_inp.shape))
        #plt.imsave("deformation.png", deformed_inp[0, 0, 0].cpu().numpy())

        return deformed_inp

    def forward(self, source_image, kp_driving, kp_source):
        appearance_skips = self.appearance_encoder(source_image)

        deformations_absolute = self.dense_motion_module(source_image=source_image, kp_driving=kp_driving,
                                                         kp_source=kp_source)

        deformed_skips = [self.deform_input(skip, deformations_absolute) for skip in appearance_skips]
        

        if self.kp_embedding_module is not None:
            d = kp_driving['mean'].shape[1]
            movement_embedding = self.kp_embedding_module(source_image=source_image, kp_driving=kp_driving,
                                                          kp_source=kp_source)
            
            print(movement_embedding.shape)
            plt.imsave("moveembed.png", movement_embedding[0, 0, 0].cpu().numpy())

            kp_skips = [F.interpolate(movement_embedding, size=(d,) + skip.shape[3:], mode=self.interpolation_mode) for skip in appearance_skips]
            skips = [torch.cat([a, b], dim=1) for a, b in zip(deformed_skips, kp_skips)]
        else:
            skips = deformed_skips
        
        """print("absolute deformations: " + str(deformations_absolute.data.shape))
        deform_abs = deformations_absolute.data[0, 0].cpu().numpy()
        plt.imsave("abs_deformation" + str(test_id) + ".png", np.sqrt(deform_abs[:, :, 0] ** 2 + deform_abs[:, :, 1] ** 2 + deform_abs[:, :, 2] ** 2))
        test_id = test_id + 1"""
        
        video_deformed = self.deform_input(source_image, deformations_absolute)
        #print(video_deformed.shape)
        #plt.imsave("deformed.png", video_deformed[0, 0, 0].cpu().numpy())
        video_prediction = self.video_decoder(skips)

        """print(video_prediction.shape)
        test = np.zeros((128, 128))
        for i in video_prediction[0].cpu().numpy():
            test += i[0]
        plt.imsave("prediction.png", test)"""

        video_prediction = self.refinement_module(video_prediction)
        print(video_prediction.shape)
        #plt.imsave("prediction.png", video_prediction[0, 0, 0].cpu().numpy())
        video_prediction = torch.sigmoid(video_prediction)

        #print(video_prediction.shape)
        plt.imsave("prediction.png", video_prediction[0, 0, 0].cpu().numpy())

        return {"video_prediction": video_prediction, "video_deformed": video_deformed}
