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


class SV248s(BaseVideoDataset):


    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):      
        root = env_settings().SV248s_dir if root is None else root
        super().__init__('SV248s', root, image_loader)

        # all folders inside the root
        self.sequence_list = self._get_sequence_list()

        # seq_id is the index of the folder inside the got10k root path
        if split is not None:
            if seq_ids is not None:
                raise ValueError('Cannot set both split_name and seq_ids.')
            if split == 'vottrain':
                sequence_list = []
                for i in self.sequence_list:
                    if int(i[3]) != 6:
                        sequence_list.append(i)
                self.sequence_list = sequence_list

            elif split == 'votval':
                sequence_list = []
                for i in self.sequence_list:
                    if int(i[3]) == 6:
                        sequence_list.append(i)
                self.sequence_list = sequence_list
            else:
                raise ValueError('Unknown split name.')


        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()

        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    def get_name(self):
        return 'sv248s'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s[2:4]+'_annotations'), '0000'+s[4:]+'.abs') for s in self.sequence_list}
        return sequence_meta_info

    def _read_meta(self, seq_path, abs_name):
        try:
            meta_info = np.loadtxt(os.path.join(seq_path, abs_name), delimiter='"', dtype=str)

            object_meta = OrderedDict({'object_class_name': str(meta_info[25]),
                                       'motion_class': None,
                                       'major_class': str(meta_info[25]),
                                       'root_class': str(meta_info[25]),
                                       'motion_adverb': str(meta_info[29])})
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
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
        with open(os.path.join(self.root, 'list.txt')) as f:
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

    def _read_bb_anno(self, seq_path):
        gt = pandas.read_csv(seq_path+'.rect', delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path+'.state')

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])

        target_visible = ~occlusion

        return target_visible
    def _get_sequence_path(self, seq_id):
        s = self.sequence_list[seq_id]
        return os.path.join(self.root, s[2:4]+'_annotations') + '/0000'+s[4:]

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0) 
        visible = self._read_target_visible(seq_path)
        visible = visible & valid.byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:06}.jpg'.format(frame_id+1))   

    def _get_frame(self, frame_id):
        return self.image_loader(frame_id)

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class_name']

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = os.path.join(self.root , self.sequence_list[seq_id])
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        frame_list_ = ['{sequence_path}/{frame:0{nz}}.jpg'.format(
            sequence_path=seq_path, frame=f_id + self._get_sequence_info_list()[seq_id]['startFrame'],
            nz=self._get_sequence_info_list()[seq_id]['nz']) for f_id in frame_ids]
        frame_list = [self._get_frame(f_id) for f_id in frame_list_]

        if anno is None:
            anno = self.get_sequence_info(seq_id)
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
        
        return frame_list, anno_frames, obj_meta

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "000100", "path": "000100", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000000.rect", "object_class": "car-large"},
            {"name": "000101", "path": "000101", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000001.rect", "object_class": "car"},
            {"name": "000102", "path": "000102", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000002.rect", "object_class": "car"},
            {"name": "000103", "path": "000103", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000003.rect", "object_class": "car-large"},
            {"name": "000104", "path": "000104", "startFrame": 1, "endFrame": 461, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000004.rect", "object_class": "car"},
            {"name": "000105", "path": "000105", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000005.rect", "object_class": "car"},
            {"name": "000106", "path": "000106", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000006.rect", "object_class": "car"},
            {"name": "000107", "path": "000107", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000007.rect", "object_class": "car"},
            {"name": "000108", "path": "000108", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000008.rect", "object_class": "car"},
            {"name": "000109", "path": "000109", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000009.rect", "object_class": "car"},
            {"name": "000110", "path": "000110", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000010.rect", "object_class": "car"},
            {"name": "000111", "path": "000111", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000011.rect", "object_class": "car"},
            {"name": "000112", "path": "000112", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000012.rect", "object_class": "car-large"},
            {"name": "000113", "path": "000113", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000013.rect", "object_class": "car"},
            {"name": "000114", "path": "000114", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000014.rect", "object_class": "ship"},
            {"name": "000115", "path": "000115", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000015.rect", "object_class": "car"},
            {"name": "000116", "path": "000116", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000016.rect", "object_class": "car"},
            {"name": "000117", "path": "000117", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000017.rect", "object_class": "car-large"},
            {"name": "000118", "path": "000118", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000018.rect", "object_class": "car"},
            {"name": "000119", "path": "000119", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000019.rect", "object_class": "car"},
            {"name": "000120", "path": "000120", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000020.rect", "object_class": "car-large"},
            {"name": "000121", "path": "000121", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000021.rect", "object_class": "car"},
            {"name": "000122", "path": "000122", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000022.rect", "object_class": "car"},
            {"name": "000123", "path": "000123", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000023.rect", "object_class": "car"},
            {"name": "000124", "path": "000124", "startFrame": 1, "endFrame": 500, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000024.rect", "object_class": "car"},
            {"name": "000125", "path": "000125", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000025.rect", "object_class": "ship"},
            {"name": "000126", "path": "000126", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000026.rect", "object_class": "car-large"},
            {"name": "000127", "path": "000127", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000027.rect", "object_class": "car"},
            {"name": "000128", "path": "000128", "startFrame": 1, "endFrame": 486, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000028.rect", "object_class": "car"},
            {"name": "000129", "path": "000129", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000029.rect", "object_class": "ship"},
            {"name": "000130", "path": "000130", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000030.rect", "object_class": "car"},
            {"name": "000131", "path": "000131", "startFrame": 1, "endFrame": 486, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000031.rect", "object_class": "car"},
            {"name": "000132", "path": "000132", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000032.rect", "object_class": "car"},
            {"name": "000133", "path": "000133", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000033.rect", "object_class": "car"},
            {"name": "000134", "path": "000134", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000034.rect", "object_class": "car"},
            {"name": "000135", "path": "000135", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000035.rect", "object_class": "car"},
            {"name": "000136", "path": "000136", "startFrame": 1, "endFrame": 300, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000036.rect", "object_class": "car"},
            {"name": "000137", "path": "000137", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000037.rect", "object_class": "car-large"},
            {"name": "000138", "path": "000138", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000038.rect", "object_class": "car-large"},
            {"name": "000139", "path": "000139", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000039.rect", "object_class": "car-large"},
            {"name": "000140", "path": "000140", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000040.rect", "object_class": "car"},
            {"name": "000141", "path": "000141", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000041.rect", "object_class": "car"},
            {"name": "000142", "path": "000142", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000042.rect", "object_class": "car"},
            {"name": "000143", "path": "000143", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000043.rect", "object_class": "car"},
            {"name": "000144", "path": "000144", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000044.rect", "object_class": "car"},
            {"name": "000145", "path": "000145", "startFrame": 1, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000045.rect", "object_class": "car"},
            {"name": "000146", "path": "000146", "startFrame": 1, "endFrame": 291, "nz": 6, "ext": "tiff",
             "anno_path": "01_annotations/000046.rect", "object_class": "car"},
            {"name": "000200", "path": "000200", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000000.rect", "object_class": "car"},
            {"name": "000201", "path": "000201", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000001.rect", "object_class": "car"},
            {"name": "000202", "path": "000202", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000002.rect", "object_class": "car"},
            {"name": "000203", "path": "000203", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000003.rect", "object_class": "car"},
            {"name": "000204", "path": "000204", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000004.rect", "object_class": "car"},
            {"name": "000205", "path": "000205", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000005.rect", "object_class": "car"},
            {"name": "000206", "path": "000206", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000006.rect", "object_class": "car"},
            {"name": "000207", "path": "000207", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000007.rect", "object_class": "car"},
            {"name": "000208", "path": "000208", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000008.rect", "object_class": "car"},
            {"name": "000209", "path": "000209", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000009.rect", "object_class": "car"},
            {"name": "000210", "path": "000210", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000010.rect", "object_class": "car"},
            {"name": "000211", "path": "000211", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000011.rect", "object_class": "car"},
            {"name": "000212", "path": "000212", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000012.rect", "object_class": "car"},
            {"name": "000213", "path": "000213", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000013.rect", "object_class": "car"},
            {"name": "000214", "path": "000214", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000014.rect", "object_class": "car"},
            {"name": "000215", "path": "000215", "startFrame": 1, "endFrame": 601, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000015.rect", "object_class": "car-large"},
            {"name": "000216", "path": "000216", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000016.rect", "object_class": "car"},
            {"name": "000217", "path": "000217", "startFrame": 1, "endFrame": 700, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000017.rect", "object_class": "car"},
            {"name": "000218", "path": "000218", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000018.rect", "object_class": "car"},
            {"name": "000220", "path": "000220", "startFrame": 1, "endFrame": 695, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000020.rect", "object_class": "car"},
            {"name": "000221", "path": "000221", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000021.rect", "object_class": "car"},
            {"name": "000222", "path": "000222", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000022.rect", "object_class": "car"},
            {"name": "000223", "path": "000223", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000023.rect", "object_class": "car"},
            {"name": "000224", "path": "000224", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000024.rect", "object_class": "car"},
            {"name": "000225", "path": "000225", "startFrame": 80, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000025.rect", "object_class": "car"},
            {"name": "000227", "path": "000227", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000027.rect", "object_class": "car"},
            {"name": "000228", "path": "000228", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000028.rect", "object_class": "car"},
            {"name": "000229", "path": "000229", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000029.rect", "object_class": "car"},
            {"name": "000230", "path": "000230", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000030.rect", "object_class": "car"},
            {"name": "000231", "path": "000231", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000031.rect", "object_class": "car"},
            {"name": "000232", "path": "000232", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000032.rect", "object_class": "car"},
            {"name": "000233", "path": "000233", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000033.rect", "object_class": "car"},
            {"name": "000234", "path": "000234", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000034.rect", "object_class": "car"},
            {"name": "000236", "path": "000236", "startFrame": 28, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000036.rect", "object_class": "car"},
            {"name": "000237", "path": "000237", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000037.rect", "object_class": "car"},
            {"name": "000238", "path": "000238", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000038.rect", "object_class": "car"},
            {"name": "000239", "path": "000239", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000039.rect", "object_class": "car"},
            {"name": "000240", "path": "000240", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000040.rect", "object_class": "car"},
            {"name": "000241", "path": "000241", "startFrame": 1, "endFrame": 750, "nz": 6, "ext": "tiff",
             "anno_path": "02_annotations/000041.rect", "object_class": "car"},
            {"name": "000300", "path": "000300", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000000.rect", "object_class": "car"},
            {"name": "000301", "path": "000301", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000001.rect", "object_class": "car"},
            {"name": "000302", "path": "000302", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000002.rect", "object_class": "car"},
            {"name": "000303", "path": "000303", "startFrame": 168, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000003.rect", "object_class": "car"},
            {"name": "000304", "path": "000304", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000004.rect", "object_class": "car"},
            {"name": "000306", "path": "000306", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000006.rect", "object_class": "car-large"},
            {"name": "000307", "path": "000307", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000007.rect", "object_class": "car-large"},
            {"name": "000308", "path": "000308", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000008.rect", "object_class": "car"},
            {"name": "000310", "path": "000310", "startFrame": 37, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000010.rect", "object_class": "car-large"},
            {"name": "000311", "path": "000311", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000011.rect", "object_class": "car-large"},
            {"name": "000312", "path": "000312", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000012.rect", "object_class": "car-large"},
            {"name": "000313", "path": "000313", "startFrame": 15, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000013.rect", "object_class": "car"},
            {"name": "000314", "path": "000314", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000014.rect", "object_class": "car-large"},
            {"name": "000315", "path": "000315", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000015.rect", "object_class": "car-large"},
            {"name": "000316", "path": "000316", "startFrame": 16, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000016.rect", "object_class": "car-large"},
            {"name": "000317", "path": "000317", "startFrame": 16, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000017.rect", "object_class": "car-large"},
            {"name": "000318", "path": "000318", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000018.rect", "object_class": "car-large"},
            {"name": "000319", "path": "000319", "startFrame": 21, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000019.rect", "object_class": "car-large"},
            {"name": "000320", "path": "000320", "startFrame": 10, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000020.rect", "object_class": "car-large"},
            {"name": "000321", "path": "000321", "startFrame": 10, "endFrame": 647, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000021.rect", "object_class": "car"},
            {"name": "000322", "path": "000322", "startFrame": 1, "endFrame": 433, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000022.rect", "object_class": "car"},
            {"name": "000323", "path": "000323", "startFrame": 1, "endFrame": 574, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000023.rect", "object_class": "car-large"},
            {"name": "000324", "path": "000324", "startFrame": 19, "endFrame": 554, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000024.rect", "object_class": "car-large"},
            {"name": "000325", "path": "000325", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000025.rect", "object_class": "car-large"},
            {"name": "000326", "path": "000326", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000026.rect", "object_class": "car-large"},
            {"name": "000327", "path": "000327", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000027.rect", "object_class": "car-large"},
            {"name": "000328", "path": "000328", "startFrame": 1, "endFrame": 663, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000028.rect", "object_class": "car-large"},
            {"name": "000329", "path": "000329", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000029.rect", "object_class": "car"},
            {"name": "000330", "path": "000330", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000030.rect", "object_class": "car"},
            {"name": "000331", "path": "000331", "startFrame": 100, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000031.rect", "object_class": "car"},
            {"name": "000332", "path": "000332", "startFrame": 25, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000032.rect", "object_class": "car"},
            {"name": "000333", "path": "000333", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000033.rect", "object_class": "car"},
            {"name": "000334", "path": "000334", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000034.rect", "object_class": "car-large"},
            {"name": "000335", "path": "000335", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000035.rect", "object_class": "car-large"},
            {"name": "000336", "path": "000336", "startFrame": 10, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000036.rect", "object_class": "car"},
            {"name": "000337", "path": "000337", "startFrame": 25, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000037.rect", "object_class": "car-large"},
            {"name": "000338", "path": "000338", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000038.rect", "object_class": "car-large"},
            {"name": "000339", "path": "000339", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000039.rect", "object_class": "car-large"},
            {"name": "000340", "path": "000340", "startFrame": 20, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "03_annotations/000040.rect", "object_class": "car-large"},
            {"name": "000400", "path": "000400", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000000.rect", "object_class": "plane"},
            {"name": "000401", "path": "000401", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000001.rect", "object_class": "plane"},
            {"name": "000402", "path": "000402", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000002.rect", "object_class": "plane"},
            {"name": "000403", "path": "000403", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000003.rect", "object_class": "plane"},
            {"name": "000404", "path": "000404", "startFrame": 31, "endFrame": 743, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000004.rect", "object_class": "car"},
            {"name": "000405", "path": "000405", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000005.rect", "object_class": "car"},
            {"name": "000406", "path": "000406", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000006.rect", "object_class": "car"},
            {"name": "000407", "path": "000407", "startFrame": 1, "endFrame": 686, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000007.rect", "object_class": "car"},
            {"name": "000408", "path": "000408", "startFrame": 1, "endFrame": 646, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000008.rect", "object_class": "car"},
            {"name": "000409", "path": "000409", "startFrame": 1, "endFrame": 589, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000009.rect", "object_class": "car"},
            {"name": "000410", "path": "000410", "startFrame": 10, "endFrame": 746, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000010.rect", "object_class": "car"},
            {"name": "000411", "path": "000411", "startFrame": 56, "endFrame": 744, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000011.rect", "object_class": "car-large"},
            {"name": "000412", "path": "000412", "startFrame": 16, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000012.rect", "object_class": "car"},
            {"name": "000413", "path": "000413", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000013.rect", "object_class": "car"},
            {"name": "000414", "path": "000414", "startFrame": 1, "endFrame": 606, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000014.rect", "object_class": "car"},
            {"name": "000415", "path": "000415", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000015.rect", "object_class": "car"},
            {"name": "000416", "path": "000416", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000016.rect", "object_class": "car"},
            {"name": "000417", "path": "000417", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000017.rect", "object_class": "car"},
            {"name": "000418", "path": "000418", "startFrame": 10, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000018.rect", "object_class": "car"},
            {"name": "000419", "path": "000419", "startFrame": 1, "endFrame": 693, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000019.rect", "object_class": "car"},
            {"name": "000420", "path": "000420", "startFrame": 1, "endFrame": 676, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000020.rect", "object_class": "car"},
            {"name": "000421", "path": "000421", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000021.rect", "object_class": "car"},
            {"name": "000422", "path": "000422", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000022.rect", "object_class": "car"},
            {"name": "000423", "path": "000423", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000023.rect", "object_class": "car"},
            {"name": "000424", "path": "000424", "startFrame": 13, "endFrame": 490, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000024.rect", "object_class": "car"},
            {"name": "000425", "path": "000425", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000025.rect", "object_class": "car"},
            {"name": "000426", "path": "000426", "startFrame": 1, "endFrame": 277, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000026.rect", "object_class": "car"},
            {"name": "000427", "path": "000427", "startFrame": 65, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000027.rect", "object_class": "car"},
            {"name": "000428", "path": "000428", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000028.rect", "object_class": "car"},
            {"name": "000429", "path": "000429", "startFrame": 1, "endFrame": 583, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000029.rect", "object_class": "car"},
            {"name": "000430", "path": "000430", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000030.rect", "object_class": "car"},
            {"name": "000431", "path": "000431", "startFrame": 1, "endFrame": 413, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000031.rect", "object_class": "car"},
            {"name": "000432", "path": "000432", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000032.rect", "object_class": "car"},
            {"name": "000433", "path": "000433", "startFrame": 1, "endFrame": 575, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000033.rect", "object_class": "car"},
            {"name": "000434", "path": "000434", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000034.rect", "object_class": "car"},
            {"name": "000435", "path": "000435", "startFrame": 1, "endFrame": 708, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000035.rect", "object_class": "car"},
            {"name": "000436", "path": "000436", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000036.rect", "object_class": "car"},
            {"name": "000437", "path": "000437", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000037.rect", "object_class": "car"},
            {"name": "000438", "path": "000438", "startFrame": 25, "endFrame": 677, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000038.rect", "object_class": "car"},
            {"name": "000439", "path": "000439", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "04_annotations/000039.rect", "object_class": "plane"},
            {"name": "000500", "path": "000500", "startFrame": 1, "endFrame": 580, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000000.rect", "object_class": "car"},
            {"name": "000501", "path": "000501", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000001.rect", "object_class": "car"},
            {"name": "000502", "path": "000502", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000002.rect", "object_class": "car"},
            {"name": "000503", "path": "000503", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000003.rect", "object_class": "car"},
            {"name": "000506", "path": "000506", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000006.rect", "object_class": "car-large"},
            {"name": "000507", "path": "000507", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000007.rect", "object_class": "car"},
            {"name": "000508", "path": "000508", "startFrame": 1, "endFrame": 747, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000008.rect", "object_class": "car"},
            {"name": "000509", "path": "000509", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000009.rect", "object_class": "car"},
            {"name": "000510", "path": "000510", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000010.rect", "object_class": "car"},
            {"name": "000511", "path": "000511", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000011.rect", "object_class": "car"},
            {"name": "000512", "path": "000512", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000012.rect", "object_class": "car"},
            {"name": "000513", "path": "000513", "startFrame": 1, "endFrame": 740, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000013.rect", "object_class": "car"},
            {"name": "000514", "path": "000514", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000014.rect", "object_class": "car"},
            {"name": "000515", "path": "000515", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000015.rect", "object_class": "car"},
            {"name": "000516", "path": "000516", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000016.rect", "object_class": "car"},
            {"name": "000517", "path": "000517", "startFrame": 1, "endFrame": 661, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000017.rect", "object_class": "car"},
            {"name": "000518", "path": "000518", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000018.rect", "object_class": "car"},
            {"name": "000519", "path": "000519", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000019.rect", "object_class": "car"},
            {"name": "000520", "path": "000520", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000020.rect", "object_class": "car"},
            {"name": "000521", "path": "000521", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000021.rect", "object_class": "car"},
            {"name": "000522", "path": "000522", "startFrame": 1, "endFrame": 632, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000022.rect", "object_class": "car"},
            {"name": "000523", "path": "000523", "startFrame": 1, "endFrame": 723, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000023.rect", "object_class": "car-large"},
            {"name": "000524", "path": "000524", "startFrame": 1, "endFrame": 672, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000024.rect", "object_class": "car"},
            {"name": "000525", "path": "000525", "startFrame": 1, "endFrame": 460, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000025.rect", "object_class": "car"},
            {"name": "000526", "path": "000526", "startFrame": 36, "endFrame": 733, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000026.rect", "object_class": "car"},
            {"name": "000527", "path": "000527", "startFrame": 1, "endFrame": 732, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000027.rect", "object_class": "car"},
            {"name": "000528", "path": "000528", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000028.rect", "object_class": "car"},
            {"name": "000529", "path": "000529", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000029.rect", "object_class": "car"},
            {"name": "000530", "path": "000530", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000030.rect", "object_class": "car"},
            {"name": "000531", "path": "000531", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000031.rect", "object_class": "car"},
            {"name": "000532", "path": "000532", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000032.rect", "object_class": "car"},
            {"name": "000533", "path": "000533", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000033.rect", "object_class": "car"},
            {"name": "000534", "path": "000534", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000034.rect", "object_class": "car"},
            {"name": "000535", "path": "000535", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000035.rect", "object_class": "car"},
            {"name": "000536", "path": "000536", "startFrame": 1, "endFrame": 748, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000036.rect", "object_class": "car"},
            {"name": "000537", "path": "000537", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000037.rect", "object_class": "car"},
            {"name": "000538", "path": "000538", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000038.rect", "object_class": "car"},
            {"name": "000539", "path": "000539", "startFrame": 1, "endFrame": 749, "nz": 6, "ext": "tiff",
             "anno_path": "05_annotations/000039.rect", "object_class": "car"},
            {"name": "000600", "path": "000600", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000000.rect", "object_class": "plane"},
            {"name": "000601", "path": "000601", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000001.rect", "object_class": "car"},
            {"name": "000602", "path": "000602", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000002.rect", "object_class": "car"},
            {"name": "000603", "path": "000603", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000003.rect", "object_class": "car"},
            {"name": "000604", "path": "000604", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000004.rect", "object_class": "car"},
            {"name": "000605", "path": "000605", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000005.rect", "object_class": "car"},
            {"name": "000607", "path": "000607", "startFrame": 14, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000007.rect", "object_class": "car"},
            {"name": "000608", "path": "000608", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000008.rect", "object_class": "car"},
            {"name": "000609", "path": "000609", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000009.rect", "object_class": "car"},
            {"name": "000610", "path": "000610", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000010.rect", "object_class": "car"},
            {"name": "000611", "path": "000611", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000011.rect", "object_class": "car"},
            {"name": "000612", "path": "000612", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000012.rect", "object_class": "car"},
            {"name": "000613", "path": "000613", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000013.rect", "object_class": "car"},
            {"name": "000614", "path": "000614", "startFrame": 25, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000014.rect", "object_class": "car"},
            {"name": "000616", "path": "000616", "startFrame": 40, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000016.rect", "object_class": "car"},
            {"name": "000617", "path": "000617", "startFrame": 30, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000017.rect", "object_class": "car"},
            {"name": "000618", "path": "000618", "startFrame": 30, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000018.rect", "object_class": "car"},
            {"name": "000619", "path": "000619", "startFrame": 10, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000019.rect", "object_class": "car"},
            {"name": "000620", "path": "000620", "startFrame": 10, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000020.rect", "object_class": "car"},
            {"name": "000621", "path": "000621", "startFrame": 10, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000021.rect", "object_class": "car"},
            {"name": "000622", "path": "000622", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000022.rect", "object_class": "car"},
            {"name": "000623", "path": "000623", "startFrame": 10, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000023.rect", "object_class": "car"},
            {"name": "000624", "path": "000624", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000024.rect", "object_class": "car"},
            {"name": "000625", "path": "000625", "startFrame": 10, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000025.rect", "object_class": "car"},
            {"name": "000626", "path": "000626", "startFrame": 11, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000026.rect", "object_class": "car"},
            {"name": "000628", "path": "000628", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000028.rect", "object_class": "car"},
            {"name": "000629", "path": "000629", "startFrame": 10, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000029.rect", "object_class": "car"},
            {"name": "000630", "path": "000630", "startFrame": 10, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000030.rect", "object_class": "car"},
            {"name": "000631", "path": "000631", "startFrame": 10, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000031.rect", "object_class": "car"},
            {"name": "000632", "path": "000632", "startFrame": 10, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000032.rect", "object_class": "car"},
            {"name": "000633", "path": "000633", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000033.rect", "object_class": "car"},
            {"name": "000634", "path": "000634", "startFrame": 25, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000034.rect", "object_class": "car"},
            {"name": "000635", "path": "000635", "startFrame": 10, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000035.rect", "object_class": "car"},
            {"name": "000636", "path": "000636", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000036.rect", "object_class": "car"},
            {"name": "000637", "path": "000637", "startFrame": 25, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000037.rect", "object_class": "car"},
            {"name": "000638", "path": "000638", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000038.rect", "object_class": "car"},
            {"name": "000639", "path": "000639", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000039.rect", "object_class": "car"},
            {"name": "000640", "path": "000640", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000040.rect", "object_class": "car"},
            {"name": "000641", "path": "000641", "startFrame": 20, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000041.rect", "object_class": "car"},
            {"name": "000642", "path": "000642", "startFrame": 10, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000042.rect", "object_class": "car"},
            {"name": "000643", "path": "000643", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000043.rect", "object_class": "car"},
            {"name": "000644", "path": "000644", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000044.rect", "object_class": "car"},
            {"name": "000645", "path": "000645", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000045.rect", "object_class": "car"},
            {"name": "000646", "path": "000646", "startFrame": 10, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000046.rect", "object_class": "car"},
            {"name": "000647", "path": "000647", "startFrame": 10, "endFrame": 499, "nz": 6, "ext": "tiff",
             "anno_path": "06_annotations/000047.rect", "object_class": "car"}
        ]

        sequence_info_list_end = []
        for index_sequence in sequence_info_list:
            list_txt = './trackingdata/SV248S'
            list_dir = np.genfromtxt(list_txt, delimiter=',', dtype=str)
            for index_list in list_dir:
                if index_sequence['name'] == index_list:
                    sequence_info_list_end.append(index_sequence)
                    break

        return sequence_info_list_end
