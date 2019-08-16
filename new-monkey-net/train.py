import sys
from tqdm import trange

import torch
from torch.utils.data import DataLoader

from logger import Logger
from modules.losses import generator_loss, discriminator_loss, generator_loss_names, discriminator_loss_names

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback


def split_kp(kp_joined, detach=False):
    if detach:
        kp_video = {k: v[:, 1:].detach() for k, v in kp_joined.items()}
        kp_appearance = {k: v[:, :1].detach() for k, v in kp_joined.items()}
    else:
        kp_video = {k: v[:, 1:] for k, v in kp_joined.items()}
        kp_appearance = {k: v[:, :1] for k, v in kp_joined.items()}
    return {'kp_driving': kp_video, 'kp_source': kp_appearance}


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params

    def forward(self, x):
        kp_joined_single = self.kp_extractor(torch.cat([x['source'], x['video']], dim=2))

        # print(x['source'].shape, x['video'].shape)
        # sys.exit()
        cat_x = {'source': torch.cat([x['source'], x['source']], dim=4),
                 'video': torch.cat([x['video'], x['video']], dim=4)}
        # print(cat_x['source'].shape, cat_x['video'].shape)
        # sys.exit()

        _, _, num_kp, _ = kp_joined_single['mean'].shape
        # print(kp_joined_single['mean'][0])
        kp_joined = {'mean': torch.cat((kp_joined_single['mean'], kp_joined_single['mean']), dim=2),
                     'var': torch.cat((kp_joined_single['var'], kp_joined_single['var']), dim=2)}
        # kp_joined['mean'][:, :, num_kp: 2*num_kp, 1] += 2
        generated = self.generator(cat_x['source'],
                                   **split_kp(kp_joined, self.train_params['detach_kp_generator']))

        video_prediction = generated['video_prediction']
        video_deformed = generated['video_deformed']

        kp_dict = split_kp(kp_joined, False)
        discriminator_maps_generated = self.discriminator(video_prediction, **kp_dict)
        discriminator_maps_real = self.discriminator(cat_x['video'], **kp_dict)
        # print(video_prediction.shape)
        # print(discriminator_maps_generated)
        # print(len(discriminator_maps_real))
        # print(x['source'].shape, x['video'].shape)
        # sys.exit()

        generated.update(kp_dict)

        losses = generator_loss(discriminator_maps_generated=discriminator_maps_generated,
                                discriminator_maps_real=discriminator_maps_real,
                                video_deformed=video_deformed,
                                loss_weights=self.train_params['loss_weights'])

        return tuple(losses) + (generated, kp_joined)


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params

    def forward(self, x, kp_joined, generated):
        cat_x = {'source': torch.cat([x['source'], x['source']], dim=4),
                 'video': torch.cat([x['video'], x['video']], dim=4)}
        # print(x['source'].shape)
        # print(kp_joined['var'].shape)
        # print(generated['video_prediction'].shape)
        # sys.exit()
        kp_dict = split_kp(kp_joined, self.train_params['detach_kp_discriminator'])
        discriminator_maps_generated = self.discriminator(generated['video_prediction'].detach(), **kp_dict)
        discriminator_maps_real = self.discriminator(cat_x['video'], **kp_dict)
        loss = discriminator_loss(discriminator_maps_generated=discriminator_maps_generated,
                                  discriminator_maps_real=discriminator_maps_real,
                                  loss_weights=self.train_params['loss_weights'])
        return loss


def train(config, generator, discriminator, kp_detector, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch, it = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector,
                                          optimizer_generator, optimizer_discriminator, optimizer_kp_detector)
    else:
        start_epoch = 0
        it = 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=start_epoch - 1)

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=4, drop_last=True)

    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    generator_full_par = DataParallelWithCallback(generator_full, device_ids=device_ids)
    discriminator_full_par = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], **train_params['log_params']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in dataloader:
                # print(len(x['source']))
                # sys.exit()
                # print(x.shape)
                out = generator_full_par(x)
                # for i in out:
                #     try:
                #         print(i.shape)
                #         print('!!!!!!!!')
                #     except:
                #         for s,v in i.item:
                #             print(s)
                #
                #         print(i['mean'].shape, i['var'].shape)
                # print(out[-2]['kp_driving']['var'].shape)
                # print(out[-2]['video_prediction'].shape)
                # print(out[-2]['video_deformed'].shape)
                # print(out[-2]['kp_source']['var'].shape)
                # print(out[-1]['var'].shape)
                # print('________________________')
                loss_values = out[:-2]
                generated = out[-2]
                kp_joined = out[-1]
                loss_values = [val.mean() for val in loss_values]
                loss = sum(loss_values)

                loss.backward(retain_graph=not train_params['detach_kp_discriminator'])
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_discriminator.zero_grad()
                if train_params['detach_kp_discriminator']:
                    optimizer_kp_detector.step()
                    optimizer_kp_detector.zero_grad()

                generator_loss_values = [val.detach().cpu().numpy() for val in loss_values]

                loss_values = discriminator_full_par(x, kp_joined, generated)
                loss_values = [val.mean() for val in loss_values]
                loss = sum(loss_values)

                loss.backward()
                optimizer_discriminator.step()
                optimizer_discriminator.zero_grad()
                if not train_params['detach_kp_discriminator']:
                    optimizer_kp_detector.step()
                    optimizer_kp_detector.zero_grad()

                discriminator_loss_values = [val.detach().cpu().numpy() for val in loss_values]

                logger.log_iter(it,
                                names=generator_loss_names(train_params['loss_weights']) + discriminator_loss_names(),
                                values=generator_loss_values + discriminator_loss_values, inp=x, out=generated)
                it += 1


            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()

            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector})
