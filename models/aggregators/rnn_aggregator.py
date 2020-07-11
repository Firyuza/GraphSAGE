import tensorflow as tf

from ..registry import AGGREGATOR
from ..builder import build_attention_layer

@AGGREGATOR.register_module
class RNNAggregator(tf.keras.layers.Layer):
    def __init__(self, cell_type, activation, use_concat, attention_layer=None):
        super(RNNAggregator, self).__init__()

        self.activation = getattr(tf.nn, activation)
        self.cell_type = cell_type
        self.use_concat = use_concat

        self.attention_layer = attention_layer
        if attention_layer is not None:
            self.attention_layer = build_attention_layer(attention_layer)

    def build(self, input_shape_cell, cell_units, input_shape_dense, output_shape_dense,
              attention_in_shape=None, attention_shared_out_shape=None, attention_out_shape=None):
        self.cell = tf.keras.layers.RNN(
            getattr(tf.keras.layers, self.cell_type)(cell_units),
            input_shape=input_shape_cell)
        self.cell.build(input_shape_cell)

        self.self_dense = tf.keras.layers.Dense(output_shape_dense, input_shape=(input_shape_dense,),
                                               name='self_dense', activation=None, use_bias=False)
        self.self_dense.build((input_shape_dense,))

        self.neigh_dense = tf.keras.layers.Dense(output_shape_dense, input_shape=(input_shape_dense,),
                                                name='neigh_dense', activation=None, use_bias=False)
        self.neigh_dense.build((input_shape_dense,))

        self.bn = tf.keras.layers.BatchNormalization()
        if self.use_concat:
            self.bn.build((None, 2 * output_shape_dense))
        else:
            self.bn.build((None, output_shape_dense))

        if self.attention_layer is not None:
            self.attention_layer.build(attention_in_shape, attention_shared_out_shape, attention_out_shape)

        super(RNNAggregator, self).build(())

        return

    def call(self, self_nodes, neigh_nodes, len_adj_nodes, training=True):
        if self.attention_layer is not None:
            self_nodes = self.attention_layer(self_nodes, neigh_nodes, training)

        neigh_cell = self.cell(neigh_nodes)

        self_nodes = self.self_dense(self_nodes)
        neigh_cell = self.neigh_dense(neigh_cell)

        if self.use_concat:
            output = tf.concat([self_nodes, neigh_cell], axis=1)
        else:
            output = tf.add_n([self_nodes, neigh_cell])

        output = self.bn(output, training=training)
        output = self.activation(output)

        return output