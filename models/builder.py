import tensorflow as tf
import tensorflow_addons as tfa
import inspect

from utils.registry import build_from_cfg
from .registry import (GRAPH, AGGREGATOR, LOSSES, OPTIMIZERS, CUSTOM_AGGREGATOR,
                       ATTENTION_LAYER, ATTENTION_MECHANISM)


def build(cfg, registry, default_args=None):
    return build_from_cfg(cfg, registry, default_args)


def build_aggregator(cfg):
    return build(cfg, AGGREGATOR)

def build_aggregator_layers(cfg):
    return build(cfg, CUSTOM_AGGREGATOR)

def build_loss(cfg):
    return build(cfg, LOSSES)

def build_accuracy(cfg, default_args=None):
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        try:
            return getattr(tf.keras.metrics, obj_type)()
        except:
            try:
                acc = getattr(tfa.metrics, obj_type)
                if len(args) > 0:
                    return acc(**args)
                else:
                    return acc()
            except:
                raise KeyError('{} is not in the builder'.format(obj_type))
    elif not inspect.isclass(obj_type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    else:
        return build(cfg, LOSSES)

def build_optimizer(cfg):
    return build(cfg, OPTIMIZERS)

def build_attention_layer(cfg):
    return build(cfg, ATTENTION_LAYER)

def build_attention_mechanism(cfg):
    return build(cfg, ATTENTION_MECHANISM)

def build_graph(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, GRAPH, dict(train_cfg=train_cfg, test_cfg=test_cfg))
