from .registry import DATASETS, DATA_LOADER
from utils.registry import build_from_cfg

def build_dataset(cfg, default_args=None):
    dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset

def build_data_loader(cfg, default_args=None):
    dataset = build_from_cfg(cfg, DATA_LOADER, default_args)

    return dataset