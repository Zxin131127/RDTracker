import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import os
import json
from tqdm import tqdm

class SatSOTDataset(BaseDataset):

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.SatSOT_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 4
        ext = 'jpg'
        start_frame = 0

        ground_truth_rect = self.sequence_list[sequence_name]['gt_rect']
        end_frame = len(ground_truth_rect)

        frames = ['{base_path}{sequence_path}'.format(base_path=self.base_path,
                  sequence_path=self.sequence_list[sequence_name]['img_names'][frame_num])
                  for frame_num in range(start_frame, end_frame)]

        return Sequence(sequence_name, frames, 'satsot', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        Dataset_listpath = './trackingdata/SatSOT/SatSOT.json'
        with open(os.path.join(Dataset_listpath), 'r') as f:
            meta_data = json.load(f)
        return meta_data
