import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod

from ..builder import build_aggregator

class BaseAggregator(tf.keras.models.Model, metaclass=ABCMeta):
    def __init__(self, depth, aggregators_shape, aggregator_type):
        super(BaseAggregator, self).__init__()

        self.depth = depth
        self.aggregators_shape = aggregators_shape
        self.aggregator_type = aggregator_type

        return

    def build(self, input_shape):
        self.aggregator_layers = []
        for k in range(self.depth):
            aggregator = build_aggregator(self.aggregator_type)
            aggregator.build(*self.aggregators_shape[k])

            self.aggregator_layers.append(aggregator)

        super(BaseAggregator, self).build(input_shape)

        return

    @abstractmethod
    def call_train(self, *args):
        pass

    @abstractmethod
    def call_test(self, *args):
        return

    def call(self, *args, training=True):
        if training:
            return self.call_train(*args)
        else:
            return self.call_test(*args)