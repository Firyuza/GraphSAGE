import tensorflow as tf

from ..registry import AGGREGATOR
from ..builder import build_attention_layer

@AGGREGATOR.register_module
class MeanAggregator(tf.keras.layers.Layer):
    def __init__(self, activation, use_concat, attention_layer=None):
        super(MeanAggregator, self).__init__()

        self.activation = getattr(tf.nn, activation)
        self.use_concat = use_concat

        self.attention_layer = attention_layer
        if attention_layer is not None:
            self.attention_layer = build_attention_layer(attention_layer)

    def build(self, input_shape, output_shape=1,
              attention_in_shape=None, attention_shared_out_shape=None, attention_out_shape=None):
        self.self_dense = tf.keras.layers.Dense(output_shape, input_shape=(input_shape,),
                                           name='self_dense', activation=None, use_bias=False)
        self.neigh_dense = tf.keras.layers.Dense(output_shape, input_shape=(input_shape,),
                                                name='neigh_dense', activation=None, use_bias=False)

        self.bn = tf.keras.layers.BatchNormalization()
        if self.use_concat:
            self.bn.build((None, 2 * output_shape))
        else:
            self.bn.build((None, output_shape))

        if self.attention_layer is not None:
            self.attention_layer.build(attention_in_shape, attention_shared_out_shape, attention_out_shape)

        super(MeanAggregator, self).build(())

        return

    def call(self, self_nodes, neigh_nodes, len_adj_nodes, training=True):
        if self.attention_layer is not None:
            self_nodes = self.attention_layer(self_nodes, neigh_nodes, training)

        mean_neigh = tf.divide(tf.reduce_sum(neigh_nodes, axis=1), tf.expand_dims(len_adj_nodes, 1))

        mean_neigh = self.neigh_dense(mean_neigh)
        self_nodes = self.self_dense(self_nodes)

        if self.use_concat:
            output = tf.concat([self_nodes, mean_neigh], axis=1)
        else:
            output = tf.add_n([self_nodes, mean_neigh])

        output = self.bn(output, training=training)
        output = self.activation(output)

        return output