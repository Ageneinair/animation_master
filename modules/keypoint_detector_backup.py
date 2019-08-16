from torch import nn
import torch
import torch.nn.functional as F
from modules.util import Hourglass, make_coordinate_grid, matrix_inverse, smallest_singular

from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import resize

def kp2gaussian(kp, spatial_size, kp_variance='matrix'):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['mean']

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())

    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape

    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)
    if kp_variance == 'matrix':
        var = kp['var']
        inv_var = matrix_inverse(var)
        shape = inv_var.shape[:number_of_leading_dimensions] + (1, 1, 2, 2)
        inv_var = inv_var.view(*shape)
        under_exp = torch.matmul(torch.matmul(mean_sub.unsqueeze(-2), inv_var), mean_sub.unsqueeze(-1))
        under_exp = under_exp.squeeze(-1).squeeze(-1)
        out = torch.exp(-0.5 * under_exp)
    elif kp_variance == 'single':
        out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp['var'])
    else:
        out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


def gaussian2kp(heatmap, kp_variance='matrix', clip_variance=None):
    """
    Extract the mean and the variance from a heatmap
    """
    shape = heatmap.shape
    #adding small eps to avoid 'nan' in variance
    heatmap = heatmap.unsqueeze(-1) + 1e-7
    grid = make_coordinate_grid(shape[3:], heatmap.type()).unsqueeze_(0).unsqueeze_(0).unsqueeze_(0)

    mean = (heatmap * grid).sum(dim=(3, 4))

    kp = {'mean': mean.permute(0, 2, 1, 3)}

    if kp_variance == 'matrix':
        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
        var = torch.matmul(mean_sub.unsqueeze(-1), mean_sub.unsqueeze(-2))
        var = var * heatmap.unsqueeze(-1)
        var = var.sum(dim=(3, 4))
        var = var.permute(0, 2, 1, 3, 4)
        if clip_variance:
            min_norm = torch.tensor(clip_variance).type(var.type())
            sg = smallest_singular(var).unsqueeze(-1)
            var = torch.max(min_norm, sg) * var / sg
        kp['var'] = var

    elif kp_variance == 'single':
        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
        var = mean_sub ** 2
        var = var * heatmap
        var = var.sum(dim=(3, 4))
        var = var.mean(dim=-1, keepdim=True)
        var = var.unsqueeze(-1)
        var = var.permute(0, 2, 1, 3, 4)
        kp['var'] = var

    return kp


class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and variance.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features, num_blocks, temperature,
                 kp_variance, scale_factor=1, clip_variance=None):
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels, out_features=num_kp,
                                   max_features=max_features, num_blocks=num_blocks)
        self.temperature = temperature
        self.kp_variance = kp_variance
        self.scale_factor = scale_factor
        self.clip_variance = clip_variance

    def forward(self, x, left_border, right_border):

        #print("x " + str(x.data.shape))
        #plt.imsave("visual/mgif_heatsource.png", x.data[0, 0, 0].cpu().numpy())
        
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=(1, self.scale_factor, self.scale_factor))

        heatmap = self.predictor(x)
        final_shape = heatmap.shape
        heatmap = heatmap.view(final_shape[0], final_shape[1], final_shape[2], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=3)
        heatmap = heatmap.view(*final_shape)

        #print("heatmap shape " + str(heatmap.data.shape))
        #print(heatmap.data[0, 0, 0])
        #for i, hmap in enumerate(heatmap.data[0].cpu().numpy()):
            #plt.imsave("visual/mgif_heat" + str(i) + ".png", hmap[0])
        """
        keypoint_map = np.zeros(shape = (64, 64)) + 255
        for i, hmap in enumerate(heatmap.data[0].cpu().numpy()):
            (x_coord, y_coord) = (0, 0)
            keypoint_map -= hmap[0]
            for line in range(hmap.shape[0]):
                for col in range(hmap.shape[1]):
                    x_coord += col * hmap[0][line, col]
                    y_coord += line * hmap[0][line, col]

            x_coord //= 64 * 64
            y_coord //= 64 * 64
            keypoint_map[int(y_coord), int(x_coord)] = 255

        plt.imsave('map.png', keypoint_map)
        """
        new_heatmap = torch.zeros(heatmap.shape)
        for i, hmap in enumerate(heatmap.data[0].cpu().numpy()):
            rhmap = resize(hmap[0], (hmap[0].shape[0], right_border - left_border), preserve_range = True)
            #plt.imsave('hmap' + str(i) + '.png', hmap[0])
            #plt.imsave('rhmap' + str(i) + '.png', rhmap)
            #print("rhmap shape " + str(rhmap.shape))
            #print("new hmap shape " + str(new_heatmap[0, i, 0, :, left_border:right_border].shape))
            new_heatmap[0, i, 0, :, left_border:right_border] = torch.from_numpy(rhmap)
            plt.imsave('newhmap' + str(i) + '.png', new_heatmap[0, i, 0].data.numpy())

        
        out = gaussian2kp(new_heatmap, self.kp_variance, self.clip_variance)

        #print("output of kp detecor " + str(out))

        return out
