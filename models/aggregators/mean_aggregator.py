import tensorflow as tf

from ..registry import AGGREGATOR
from ..builder import build_attention_layer

@AGGREGATOR.register_module
class MeanAggregator(tf.keras.layers.Layer):
    def __init__(self, activation, attention_layer=None):
        super(MeanAggregator, self).__init__()

        self.activation = getattr(tf.nn, activation)

        self.attention_layer = attention_layer
        if attention_layer is not None:
            self.attention_layer = build_attention_layer(attention_layer)

    def build(self, input_shape, output_shape=1,
              attention_in_shape=None, attention_shared_out_shape=None, attention_out_shape=None):
        self.dense = tf.keras.layers.Dense(output_shape, input_shape=(input_shape,),
                                           name='layer_dense', activation=None)
        self.dense.build((input_shape,))

        self.bn = tf.keras.layers.BatchNormalization()
        self.bn.build((None, output_shape))

        if self.attention_layer is not None:
            self.attention_layer.build(attention_in_shape, attention_shared_out_shape, attention_out_shape)

        super(MeanAggregator, self).build(())

        return

    def call(self, self_nodes, neigh_nodes, len_adj_nodes, training=True):
        if self.attention_layer is not None:
            self_nodes = self.attention_layer(self_nodes, neigh_nodes, training)

        mean_neigh = tf.divide(tf.reduce_sum(neigh_nodes, axis=1), tf.expand_dims(len_adj_nodes, 1))

        concat = tf.concat([self_nodes, mean_neigh], axis=1)

        output = self.dense(concat)
        output = self.bn(output, training=training)
        output = self.activation(output)

        return output