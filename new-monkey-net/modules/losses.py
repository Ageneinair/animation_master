import torch
import sys

def mean_batch(val):
    return val.view(val.shape[0], -1).mean(-1)


def reconstruction_loss(prediction, target, weight):
    if weight == 0:
        return 0
    return weight * mean_batch(torch.abs(prediction - target))


def generator_gan_loss(discriminator_maps_generated, weight):
    # print(len(discriminator_maps_generated[0][0])) ## shape(6,16,3,1)
    # sys.exit()

    scores_generated = discriminator_maps_generated[-1]
    score = (1 - scores_generated) ** 2
    return weight * mean_batch(score)


def discriminator_gan_loss(discriminator_maps_generated, discriminator_maps_real, weight):
    scores_real = discriminator_maps_real[-1]
    scores_generated = discriminator_maps_generated[-1]
    score = (1 - scores_real) ** 2 + scores_generated ** 2
    return weight * mean_batch(score)


def generator_loss_names(loss_weights):
    loss_names = []
    if loss_weights['reconstruction_deformed'] != 0:
        loss_names.append("rec_def")

    if loss_weights['reconstruction'] is not None:
        for i, _ in enumerate(loss_weights['reconstruction']):
            if loss_weights['reconstruction'][i] == 0:
                continue
            loss_names.append("layer-%s_rec" % i)

    loss_names.append("gen_gan")
    return loss_names


def discriminator_loss_names():
    return ['disc_gan']


def generator_loss(discriminator_maps_generated, discriminator_maps_real, video_deformed, loss_weights):
    loss_values = []
    if loss_weights['reconstruction_deformed'] != 0:
        loss_values.append(reconstruction_loss(discriminator_maps_real[0], video_deformed,
                                               loss_weights['reconstruction_deformed']))
    # print(len(discriminator_maps_generated[0])) ## shape(6,16,3,1)
    # print(zip(discriminator_maps_real[:-1], discriminator_maps_generated[:-1]).data)
    # sys.exit()

    if loss_weights['reconstruction'] != 0:
        for i, (a, b) in enumerate(zip(discriminator_maps_real[:-1], discriminator_maps_generated[:-1])):
            if loss_weights['reconstruction'][i] == 0:
                continue
            # print(a.shape) #[16, 3, 1, 64, 64]
            # print(b.shape) #[16, 3, 1, 64, 64]
            # print('_____')
            # sys.exit()
            loss_values.append(reconstruction_loss(b, a, weight=loss_weights['reconstruction'][i]))

    # print(len(loss_values)) # 5
    # sys.exit()
    loss_values.append(generator_gan_loss(discriminator_maps_generated, weight=loss_weights['generator_gan']))

    # print(loss_values[-1].shape) # [16]
    # sys.exit()

    return loss_values


def discriminator_loss(discriminator_maps_generated, discriminator_maps_real, loss_weights):
    loss_values = [discriminator_gan_loss(discriminator_maps_generated, discriminator_maps_real,
                                          loss_weights['discriminator_gan'])]

    return loss_values
