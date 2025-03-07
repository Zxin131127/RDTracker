import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader
from ltr.admin.environment import env_settings

class SkySat(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):

        root = env_settings().SkySat_dir if root is None else root
        super().__init__('SkySat', root, image_loader)

        # all folders inside the root
        self.sequence_list = self._get_sequence_list()

        # seq_id is the index of the folder inside the got10k root path
        if split is not None:
            if seq_ids is not None:
                raise ValueError('Cannot set both split_name and seq_ids.')
            if split == 'vottrain':
                sequence_list = []
                for i in range(0,len(self.sequence_list)):
                    if i < 120:
                        sequence_list.append(self.sequence_list[i])
                self.sequence_list = sequence_list

            elif split == 'votval':
                sequence_list = []
                for i in range(0,len(self.sequence_list)):
                    if i >= 120:
                        sequence_list.append(self.sequence_list[i])
                self.sequence_list = sequence_list
            else:
                raise ValueError('Unknown split name.')


        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()

        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()
        print('--------- successed SkySat ------------')

    def get_name(self):
        return 'skysat'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(s) for s in self.sequence_list}
        return sequence_meta_info

    def _read_meta(self, abs_name):
        object_meta = OrderedDict({'object_class_name': 'car',
                                       'motion_class': None,
                                       'major_class': 'car',
                                       'root_class': 'car',
                                       'motion_adverb': None})
        return object_meta

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self):
        # with open(os.path.join(self.root, 'list.txt')) as f:
        with open(os.path.join('/data/zxin/trackingdata/SkySat/list.txt')) as f:
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

    def _read_bb_anno(self, seq_path):
        # bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(seq_path, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        seq_path = os.path.join(self.root, self.sequence_list[seq_id], 'groundtruth.txt')
        bbox = self._read_bb_anno(seq_path)
        img_path = self._get_frame_path(os.path.join(self.root, self.sequence_list[seq_id], 'img'),1)
        img = self._get_frame(img_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0) & (bbox[:, 0] < img.shape[1]) & (bbox[:, 1] < img.shape[0])

        visible = valid.byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):

        return os.path.join(seq_path, '{:04}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, frame_id):
        return self.image_loader(frame_id)

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class_name']

    def get_frames(self, seq_id, frame_ids, anno=None):
        # anno_path = self._get_sequence_path(seq_id)
        seq_path = os.path.join(self.root , self.sequence_list[seq_id], 'img')
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        frame_list = [self._get_frame(os.path.join(seq_path, '{:04}.jpg'.format(f_id+1))) for f_id in frame_ids]
        # frame_list = [self._get_frame(frame_list[f_id]) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
        frame_cut = []
        bbox_cut = []
        for i in range(0, len(frame_list)):
            bbox_i = anno_frames['bbox'][i]
            frame_i = frame_list[i]
            bbox_min = max(bbox_i[2], bbox_i[3], 5)
            x1 = max(int(bbox_i[0] + bbox_i[2] / 2 - bbox_min * 3.5), 1)
            x2 = min(int(bbox_i[0] + bbox_i[2] / 2 + bbox_min * 3.5), frame_i.shape[1])
            y1 = max(int(bbox_i[1] + bbox_i[3] / 2 - bbox_min * 3.5), 1)
            y2 = min(int(bbox_i[1] + bbox_i[3] / 2 + bbox_min * 3.5), frame_i.shape[0])
            frame_new = frame_i[y1:y2, x1:x2, :]
            if frame_new.shape[0] == 0 or frame_new.shape[1] == 0:
                print('--- satsot error ! ---')
                print(self.sequence_list[seq_id]['img_names'])
                raise TypeError
            bbox_new = bbox_i - torch.tensor((x1, y1, 0, 0))

            frame_cut.append(frame_new)
            bbox_cut.append(bbox_new)
        anno_frames['bbox'] = bbox_cut
        return frame_cut, anno_frames, obj_meta
        # return frame_list, anno_frames, obj_meta
