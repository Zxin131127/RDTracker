import torch.optim as optim
import torch.nn as nn
import torch
from ltr.dataset import Lasot, Got10k, TrackingNet, MSCOCOSeq, SV248s, VISO, SatSOT, SkySat
from ltr.data import processing, sampler, LTRLoader
from ltr.models.tracking import dimpnet
import ltr.models.loss as ltr_losses
import ltr.models.loss.kl_regression as klreg_losses
import ltr.actors.tracking as tracking_actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
from ltr.data.image_loader import opencv_loader
import os

def run(settings):
    settings.description = 'Transformer-assisted tracker. Our baseline approach is SuperDiMP'
    settings.batch_size = 20 #### 20
    settings.num_workers = 8
    settings.multi_gpu = True
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 6.0#6.0
    settings.output_sigma_factor = 1/4
    settings.target_filter_sz = 4
    settings.feature_sz = 22
    settings.output_sz = settings.feature_sz * 16
    settings.center_jitter_factor = {'train': 3, 'test': 5.5}
    settings.scale_jitter_factor = {'train': 0.25, 'test': 0.5}
    settings.hinge_threshold = 0.05
    # settings.print_stats = ['Loss/total', 'Loss/iou', 'ClfTrain/init_loss', 'ClfTrain/test_loss']

    # -----------------------------------------------
    #        Train datasets
    # --- satellite dataset train ---
    if settings.test_video == 'sv248s':
        satsot_train = SatSOT(settings.env.satsot_dir, image_loader=opencv_loader, split='vottrain')
        satsot_val = SatSOT(settings.env.satsot_dir, image_loader=opencv_loader, split='votval')
    elif settings.test_video == 'viso':
        satsot_train = SatSOT(settings.env.satsot_dir, image_loader=opencv_loader, split='vottrain')
        satsot_val = SatSOT(settings.env.satsot_dir, image_loader=opencv_loader, split='votval')
        sv248s_train = SV248s(settings.env.sv248s_dir, image_loader=opencv_loader, split='vottrain')
        sv248s_val = SV248s(settings.env.sv248s_dir, image_loader=opencv_loader, split='votval')
    elif settings.test_video == 'satsot':
        sv248s_train = SV248s(settings.env.sv248s_dir, image_loader=opencv_loader, split='vottrain')
        sv248s_val = SV248s(settings.env.sv248s_dir, image_loader=opencv_loader, split='votval')
    viso_train = VISO(settings.env.viso_dir, image_loader=opencv_loader, split='vottrain')
    viso_val = VISO(settings.env.viso_dir, image_loader=opencv_loader, split='votval')
    skysat_train = SkySat(settings.env.SkySat_dir, image_loader=opencv_loader, split='vottrain')
    skysat_val = SkySat(settings.env.SkySat_dir, image_loader=opencv_loader, split='votval')

    # --- checkpoints/ltr/dimp dir ---
    # got10k_train = Got10k(settings.env.got10k_dir, image_loader=opencv_loader, split='vottrain')
    # coco_train = MSCOCOSeq(settings.env.coco_dir, image_loader=opencv_loader)

    # --old
    # coco_train = MSCOCOSeq(settings.env.coco_dir)
    # trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(4)))
    # lasot_train = Lasot(settings.env.lasot_dir, split='train')

    # -----------------------------------------------
    #      Validation datasets
    # --- satellite dataset train ---
    # sv248s_val = SV248s(settings.env.sv248s_dir, image_loader=opencv_loader, split='votval')
    # viso_val = VISO(settings.env.viso_dir, image_loader=opencv_loader, split='votval')

    # --- checkpoints/ltr/dimp dir ---
    # got10k_val = Got10k(settings.env.got10k_dir, image_loader=opencv_loader, split='votval')

    # --old
    # got10k_val = Got10k(settings.env.got10k_dir, split='votval')

    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip(probability=0.5),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # The tracking pairs processing module
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    proposal_params = {'boxes_per_frame': 128, 'gt_sigma': (0.05, 0.05), 'proposal_sigma': [(0.05, 0.05), (0.5, 0.5)]}
    label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz}
    label_density_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz}

    # Create network and actor
    net = dimpnet.dimpnet50(filter_size=settings.target_filter_sz, backbone_pretrained=True, optim_iter=5,
                            clf_feat_norm=True, clf_feat_blocks=0, final_conv=True, out_feature_dim=512,
                            optim_init_step=0.9, optim_init_reg=0.1,
                            init_gauss_sigma=output_sigma * settings.feature_sz, num_dist_bins=100,
                            bin_displacement=0.1, mask_init_factor=3.0, target_mask_act='sigmoid', score_act='relu',
                            frozen_backbone_layers=['conv1', 'bn1', 'layer1', 'layer2'],
                            qxm = settings.qxm)# not change the weight
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)#, device_ids=[1])
            # net = nn.DistributedDataParallel(net, device_ids=[0,1,3])
            # net.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        # net = MultiGPU(net, dim=1)
    data_processing_train = processing.KLDiMPProcessing(search_area_factor=settings.search_area_factor,
                                                        output_sz=settings.output_sz,
                                                        center_jitter_factor=settings.center_jitter_factor,
                                                        scale_jitter_factor=settings.scale_jitter_factor,
                                                        crop_type='inside_major',
                                                        max_scale_change=1.5,
                                                        mode='sequence',
                                                        proposal_params=proposal_params,
                                                        label_function_params=label_params,
                                                        label_density_params=label_density_params,
                                                        transform=transform_train,
                                                        joint_transform=transform_joint)

    data_processing_val = processing.KLDiMPProcessing(search_area_factor=settings.search_area_factor,
                                                      output_sz=settings.output_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      crop_type='inside_major',
                                                      max_scale_change=1.5,
                                                      mode='sequence',
                                                      proposal_params=proposal_params,
                                                      label_function_params=label_params,
                                                      label_density_params=label_density_params,
                                                      transform=transform_val,
                                                      joint_transform=transform_joint)

    # Train sampler and loader
    # dataset_train = sampler.DiMPSampler([lasot_train, got10k_train, trackingnet_train, coco_train], [1,1,1,1],
    #                                     samples_per_epoch=50000, max_gap=500, num_test_frames=3, num_train_frames=3,## 40000, 200
    #                                     processing=data_processing_train)
    #--- nature dataset
    # dataset_train = sampler.DiMPSampler([got10k_train,coco_train], [1,1],
    #                                     samples_per_epoch=50000, max_gap=500, num_test_frames=3, num_train_frames=3,## 40000, 200
    #                                     processing=data_processing_train)

    if settings.test_video == 'sv248s':
        dataset_train = sampler.DiMPSampler([viso_train, satsot_train, skysat_train], [1,1,1],
                                            samples_per_epoch=50000, max_gap=500, num_test_frames=3, num_train_frames=3,
                                            ## 40000, 200
                                            processing=data_processing_train)
    elif settings.test_video == 'viso':
        dataset_train = sampler.DiMPSampler([viso_train, satsot_train, sv248s_train, skysat_train], [1,1,1,1],
                                            samples_per_epoch=50000, max_gap=500, num_test_frames=3, num_train_frames=3,
                                            ## 40000, 200
                                            processing=data_processing_train)
    elif settings.test_video == 'satsot':
        dataset_train = sampler.DiMPSampler([viso_train, sv248s_train, skysat_train], [1,1,1],
                                            samples_per_epoch=50000, max_gap=500, num_test_frames=3, num_train_frames=3,
                                            ## 40000, 200
                                            processing=data_processing_train)


    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    # Validation samplers and loaders
    #--- nature dataset
    # dataset_val = sampler.DiMPSampler([got10k_val], [1], samples_per_epoch=10000, max_gap=500,## 200
    #                                   num_test_frames=3, num_train_frames=3,
    #                                   processing=data_processing_val)

    if settings.test_video == 'sv248s':
        dataset_val = sampler.DiMPSampler([viso_val,skysat_val,satsot_val], [1,1,1], samples_per_epoch=10000, max_gap=500,  ## 200
                                          num_test_frames=3, num_train_frames=3,
                                          processing=data_processing_val)
    elif settings.test_video == 'viso':
        dataset_val = sampler.DiMPSampler([viso_val, sv248s_val,skysat_val,satsot_val], [1,1,1,1], samples_per_epoch=10000, max_gap=500,  ## 200
                                          num_test_frames=3, num_train_frames=3,
                                          processing=data_processing_val)
    elif settings.test_video == 'satsot':
        dataset_val = sampler.DiMPSampler([viso_val, sv248s_val,skysat_val], [1,1,1], samples_per_epoch=10000, max_gap=500,  ## 200
                                          num_test_frames=3, num_train_frames=3,
                                          processing=data_processing_val)

    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=5, stack_dim=1)

    objective = {'bb_ce': klreg_losses.KLRegression(), 'test_clf': ltr_losses.LBHinge(threshold=settings.hinge_threshold)}

    loss_weight = {'bb_ce': 0.01, 'test_clf': 100, 'test_init_clf': 100, 'test_iter_clf': 400}

    actor = tracking_actors.KLDiMPActor(net=net, objective=objective, loss_weight=loss_weight)

    # Optimizer
    # optimizer = optim.Adam([{'params': actor.net.classifier.filter_initializer.parameters(), 'lr': 5e-5},
    #                         {'params': actor.net.classifier.filter_optimizer.parameters(), 'lr': 5e-4},
    #                         {'params': actor.net.classifier.feature_extractor.parameters(), 'lr': 5e-5},
    #                         {'params': actor.net.classifier.transformer.parameters(), 'lr': 1e-3},#################
    #                         {'params': actor.net.bb_regressor.parameters(), 'lr': 1e-3},
    #                         {'params': actor.net.feature_extractor.layer3.parameters(), 'lr': 2e-5}],
    #                        lr=2e-4)
    optimizer = optim.Adam([{'params': actor.net.module.classifier.filter_initializer.parameters(), 'lr': 5e-5},
                            {'params': actor.net.module.classifier.filter_optimizer.parameters(), 'lr': 5e-4},
                            {'params': actor.net.module.classifier.feature_extractor.parameters(), 'lr': 5e-5},
                            {'params': actor.net.module.classifier.transformer.parameters(), 'lr': 1e-3},  #################
                            {'params': actor.net.module.bb_regressor.parameters(), 'lr': 1e-3},
                            {'params': actor.net.module.feature_extractor.layer3.parameters(), 'lr': 2e-5}],
                           lr=2e-4)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)# self.actor.to(device)
    # trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler)

    trainer.train(50, load_latest=True, fail_safe=True)
