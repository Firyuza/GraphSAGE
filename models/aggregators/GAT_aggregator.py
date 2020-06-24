import tensorflow as tf

from ..registry import AGGREGATOR
from ..builder import build_attention_layer

@AGGREGATOR.register_module
class GATAggregator(tf.keras.layers.Layer):
    def __init__(self, activation, attention_layer=None):
        super(GATAggregator, self).__init__()

        self.activation = getattr(tf.nn, activation)

        self.attention_layer = build_attention_layer(attention_layer)

    def build(self, attention_in_shape, attention_shared_out_shape, attention_out_shape):

        self.attention_layer.build(attention_in_shape, attention_shared_out_shape, attention_out_shape)

        super(GATAggregator, self).build(())

        return

    def call(self, self_nodes, neigh_nodes, len_adj_nodes, training=True):
        self_nodes = self.attention_layer(self_nodes, neigh_nodes, training)

        return self_nodes