import logging
import random
import re
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from .runner import Runner
from datasets.builder import build_data_loader
from models.builder import build_optimizer


def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger('mmdet')
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


def set_random_seed(seed):
    """Set random seed.

    Args:
        seed (int): Seed to be used..
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def batch_processor(model, data, train_mode):
    """
    Process a data batch.

    :param model:
    :param data:
    :param train_mode:
    :return:
    """
    results, vis_embeddings, batch_labels, predictions = model(*data, train_mode=train_mode)

    print_str = ''
    for key, value in results.items():
        if key == 'output':
            continue
        print_str += '%s: %f\t' % (key, value)

    print(print_str)

    return results, vis_embeddings, batch_labels, predictions

def train_model(model,
                    dataset,
                    cfg,
                    validate=False,
                    logger=None,
                    timestamp=None):
    logger = get_root_logger(cfg.log_level)

    if validate:
        raise NotImplementedError('Built-in validation is not implemented '
                                  'yet in not-distributed training. Use '
                                  'distributed training or test.py and '
                                  '*eval.py scripts instead.')
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_data_loader(cfg.data_loader.train,
                          default_args={'dataset': ds})
        for ds in dataset
    ]

    # build TF model
    model.build()
    model.summary()

    # create optimizer
    optimizer = build_optimizer(cfg.optimizer)

    # build runner
    runner = Runner(
        model, batch_processor, optimizer, cfg.work_dir, logger=logger)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    if cfg.restore_model_path:
        runner.load_checkpoint(cfg.restore_model_path)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
