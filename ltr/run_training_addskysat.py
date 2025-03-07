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


def run_training_(train_module, train_name, qxm, train_dir, save_name, test_video, cudnn_benchmark=True):
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
    settings.qxm = qxm
    settings.test_video = test_video
    # settings.project_path = 'ltr/train_settings/{}/{}'.format(train_module, train_name)
    settings.project_path = 'ltr/{}/{}'.format(train_dir, save_name)

    expr_module = importlib.import_module('ltr.train_settings.{}.{}'.format(train_module, train_name))
    expr_func = getattr(expr_module, 'run')

    expr_func(settings)


def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    # parser.add_argument('qxm_1', type=float)
    # parser.add_argument('qxm_2', type=float)
    # parser.add_argument('train_dir', type=str)
    # parser.add_argument('save_name', type=str)
    # parser.add_argument('test_video', type=str)

    parser.add_argument('--qxm_1', type=float, default=0)
    parser.add_argument('--qxm_2', type=float, default= 0.5)
    parser.add_argument('--train_dir', type=str, default='test_satsot')
    parser.add_argument('--save_name', type=str, default='qxm_0_50_selfattention_skysat')
    parser.add_argument('--test_video', type=str, default='satsot')

    parser.add_argument('--debug', type=int, default=1, help='Debug level.')
    parser.add_argument('--train_module', type=str, default='dimp', help='Name of module in the "train_settings/" folder.')
    parser.add_argument('--train_name', type=str, default='transformer_dimp_test', help='Name of the train settings file.')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')

    args = parser.parse_args()
    args.qxm = [args.qxm_1, args.qxm_2]
    del args.qxm_1
    del args.qxm_2

    run_training_(args.train_module, args.train_name, args.qxm, args.train_dir, args.save_name, args.test_video, args.cudnn_benchmark)
    # run_training(args.train_module, args.train_name, args.cudnn_benchmark)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
    multiprocessing.set_start_method('spawn', force=True)
    main()
