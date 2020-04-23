from utils.registry import build_from_cfg
from .registry import (GRAPH, AGGREGATOR, LOSSES, OPTIMIZERS, CUSTOM_AGGREGATOR)


def build(cfg, registry, default_args=None):
    return build_from_cfg(cfg, registry, default_args)


def build_aggregator(cfg):
    return build(cfg, AGGREGATOR)

def build_custom_aggregator(cfg):
    return build(cfg, CUSTOM_AGGREGATOR)

def build_loss(cfg):
    return build(cfg, LOSSES)

def build_optimizer(cfg):
    return build(cfg, OPTIMIZERS)

def build_graph(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, GRAPH, dict(train_cfg=train_cfg, test_cfg=test_cfg))
