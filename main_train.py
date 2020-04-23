import argparse
import os
import os.path as osp
import time
import tensorflow as tf

from utils.config import Config
from utils.registry import build_from_cfg
from datasets.builder import build_dataset
from models.builder import build_graph
from API.train import train_model, set_random_seed, get_root_logger

print("GPU Available: ", tf.test.is_gpu_available())
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default='/home/firiuza/PycharmProjects/GraphSAGE/configs/graphSAGE_config.py',
                        help='train config file path')
    parser.add_argument('--work_dir', default='', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', default='', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        default='',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=555, help='random seed')
    parser.add_argument(
        '--deterministic',
        default='',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        default='',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # create work_dir
    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir)
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # log some basic info
    logger.info('Config:\n{}'.format(cfg.text))

    # set random seeds
    if args.seed is not None:
        # logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.valid))

    model = build_graph(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    train_model(model, datasets, cfg)