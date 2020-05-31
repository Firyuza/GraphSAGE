import os.path as osp

import numpy as np

from .registry import DATASETS
from utils.fileio.file_loader import load


@DATASETS.register_module
class CustomDataset:
    CLASSES = None

    def __init__(self,
                 dataset_name,
                 ann_file,
                 data_root=None,
                 test_mode=False,
                 depth=2,
                 nrof_neigh_per_batch=10):
        self.ann_file = ann_file
        self.data_root = data_root
        self.test_mode = test_mode
        self.dataset_name = dataset_name

        self.depth = depth
        self.nrof_neigh_per_batch = nrof_neigh_per_batch

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)

        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        return load(ann_file)

    def load_proposals(self, proposal_file):
        return load(proposal_file)

    def prepare_train_indices(self):
        raise NotImplementedError
