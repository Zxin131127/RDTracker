import os
import sys
import argparse
import importlib
import multiprocessing
import cv2 as cv
import torch.backends.cudnn

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import ltr.admin.settings as ws_settings


def run_training_(train_module, train_name, train_dir, yaml_type, channel_number, transformer_type, cascade_level, pre_dim, conv2_dim, optimize_filter,save_name, cudnn_benchmark=True):
    """Run a train scripts in train_settings.
    args:
        train_module: Name of module in the "train_settings/" folder.
        train_name: Name of the train settings file.
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """

    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    print('Training:  {}  {}'.format(train_module, train_name))

    settings = ws_settings.Settings()
    settings.module_name = train_module
    settings.script_name = train_name
    settings.yaml_type = yaml_type
    settings.optimize_filter = optimize_filter
    settings.pre_dim = pre_dim
    settings.transformer_type = transformer_type
    settings.conv2_dim = conv2_dim
    settings.channel_number = channel_number
    settings.cascade_level =cascade_level
    # settings.project_path = 'ltr/train_settings/{}/{}'.format(train_module, train_name)
    settings.project_path = 'ltr/{}/{}'.format(train_dir, save_name)

    expr_module = importlib.import_module('ltr.train_settings.{}.{}'.format(train_module, train_name))
    expr_func = getattr(expr_module, 'run')

    expr_func(settings)


def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    # parser.add_argument('yaml_type', type=str)
    # parser.add_argument('transformer_type', type=str)# 'no_adaptive' 'adaptive' 'adaptive_three'



    # parser.add_argument('save_name', type=str)
    # parser.add_argument('--save_name', type=str, default='qxm_adaptive_mvit_T_3reg')
    # parser.add_argument('use_net', type=str)  # 'resnet' 'swin' 'mvit'
    parser.add_argument('--use_net', type=str, default='mvit')# 'resnet' 'swin' 'mvit'
    parser.add_argument('--yaml_type', type=str, default='T')# mvit: 'T' 'S' 'B' 'L'
    parser.add_argument('--transformer_type', type=str, default='adaptive_two_add')# 'no_adaptive' 'adaptive' 'adaptive_three' 'adaptive_three_add'
    parser.add_argument('--cascade_level', type=int, default=1)# 1 2,3,4,6
    parser.add_argument('--channel_number', type=int, default=23)# 123  23
    parser.add_argument('--pre_dim', type=float,default=1)  # 0.5  pred_input_dim= [ feature_dim //2 ] --> MultiThreeIoUNet
                                                            # 1    pred_input_dim= [  feature_dim ]     --> MultiThreeIoUNet_B0 or MultiThreeIoUNet_B0_large
    parser.add_argument('--conv2_dim', type=float, default=0.25)  # 0.25  MultiThreeIoUNet or MultiThreeIoUNet_B0
                                                                 # 0.5   MultiThreeIoUNet_B0_large

    parser.add_argument('--optimize_filter', type=str, default='SDGN') # SDGN CGGN
    parser.add_argument('--train_dir', type=str, default='skysat_viso_coco_lasot_got_trackingnet')
    parser.add_argument('--debug', type=int, default=1, help='Debug level.')
    parser.add_argument('--train_module', type=str, default='dimp', help='Name of module in the "train_settings/" folder.')
    parser.add_argument('--train_name', type=str, default='transformer_dimp', help='Name of the train settings file.')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')

    args = parser.parse_args()
    # args.save_name = 'qxm_adaptive_mvit_' + args.yaml_type + '_' + str(args.channel_number)

    if args.channel_number == 23:
        args.save_name = 'qxm_' + args.transformer_type + '_' + args.use_net + '_' + args.yaml_type + '_' + str(args.channel_number)
    elif args.channel_number == 123:
        args.save_name = 'qxm_' + args.transformer_type + '_' + args.use_net + '_' + args.yaml_type + '_' + str(args.channel_number) + '_' + str(args.pre_dim) + 'predim_' + str(args.conv2_dim) + 'conv2dim'
    print('=== output model path: {} ==='.format(args.save_name))
    run_training_(args.train_module, args.train_name,     args.train_dir, #args.use_net,
                  args.yaml_type,    args.channel_number, args.transformer_type, args.cascade_level, args.pre_dim,         args.conv2_dim,  args.optimize_filter,
                  args.save_name,    args.cudnn_benchmark)
    # run_training(args.train_module, args.train_name, args.cudnn_benchmark)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    multiprocessing.set_start_method('spawn', force=True)
    main()
