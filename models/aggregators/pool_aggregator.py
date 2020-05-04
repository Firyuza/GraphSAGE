import tensorflow as tf

from ..registry import AGGREGATOR
from ..builder import build_attention_layer

@AGGREGATOR.register_module
class PoolAggregator(tf.keras.layers.Layer):
    def __init__(self, activation, pool_op, attention_layer):
        super(PoolAggregator, self).__init__()

        self.activation = getattr(tf.nn, activation)
        self.pool_op = pool_op

        self.attention_layer = attention_layer
        if attention_layer is not None:
            self.attention_layer = build_attention_layer(attention_layer)

    def build(self, input_shape, transform_output_shape, dense_input_shape, output_shape=1,
              attention_in_shape=None, attention_out_shape=None):
        self.transform_node_weight = tf.keras.layers.Dense(transform_output_shape, input_shape=(input_shape,),
                                           name='transform_node_weight', activation=None)
        self.transform_node_weight.build((input_shape, ))

        self.dense_out = tf.keras.layers.Dense(output_shape, input_shape=(dense_input_shape,),
                                           name='layer_dense', activation=self.activation)
        self.dense_out.build((dense_input_shape,))

        if self.attention_layer is not None:
            self.attention_layer.build(attention_in_shape, attention_out_shape)

        super(PoolAggregator, self).build(())

        return

    def call(self, self_nodes, neigh_nodes, len_adj_nodes, training=True):
        if self.attention_layer is not None:
            coefficients = self.attention_layer(self_nodes, neigh_nodes)
            self_nodes = tf.nn.leaky_relu(tf.reduce_sum(tf.multiply(coefficients, neigh_nodes), axis=1))

        neigh = self.transform_node_weight(neigh_nodes)
        neigh = getattr(tf, self.pool_op)(neigh, axis=1)

        concat = tf.concat([self_nodes, neigh], axis=1)
        concat = self.activation(concat)

        output = self.dense_out(concat)

        return output