import tensorflow as tf

from ..registry import AGGREGATOR
from ..builder import build_attention_layer

@AGGREGATOR.register_module
class RNNAggregator(tf.keras.layers.Layer):
    def __init__(self, cell_type, activation, attention_layer=None):
        super(RNNAggregator, self).__init__()

        self.activation = getattr(tf.nn, activation)
        self.cell_type = cell_type

        self.attention_layer = attention_layer
        if attention_layer is not None:
            self.attention_layer = build_attention_layer(attention_layer)

    def build(self, input_shape_cell, cell_units, input_shape_dense, output_shape_dense,
              attention_in_shape=None, attention_out_shape=None):
        self.cell = tf.keras.layers.RNN(
            getattr(tf.keras.layers, self.cell_type)(cell_units),
            input_shape=input_shape_cell)
        self.cell.build(input_shape_cell)

        self.dense_out = tf.keras.layers.Dense(output_shape_dense, input_shape=(input_shape_dense,),
                                               name='layer_dense', activation=self.activation)
        self.dense_out.build((input_shape_dense,))

        if self.attention_layer is not None:
            self.attention_layer.build(attention_in_shape, attention_out_shape)

        super(RNNAggregator, self).build(())

        return

    def call(self, self_nodes, neigh_nodes, len_adj_nodes, training=True):
        if self.attention_layer is not None:
            coefficients = self.attention_layer(self_nodes, neigh_nodes)
            self_nodes = tf.nn.leaky_relu(tf.reduce_sum(tf.multiply(coefficients, neigh_nodes), axis=1))

        neigh_cell = self.cell(neigh_nodes)

        concat = tf.concat([self_nodes, neigh_cell], axis=1)

        output = self.dense_out(concat)

        return output