import tensorflow as tf

from ..registry import AGGREGATOR

@AGGREGATOR.register_module
class RNNAggregator(tf.keras.layers.Layer):
    def __init__(self, cell_type, activation):
        super(RNNAggregator, self).__init__()

        self.activation = getattr(tf.nn, activation)
        self.cell_type = cell_type


    def build(self, input_shape_cell, cell_units, input_shape_dense, output_shape_dense):
        # self.cell = getattr(tf.keras.layers, self.cell_type)(cell_units,
        #                                                      input_shape=input_shape_cell)
        self.cell = tf.keras.layers.RNN(
            getattr(tf.keras.layers, self.cell_type)(cell_units),
            input_shape=input_shape_cell)
        self.cell.build(input_shape_cell)

        self.dense_out = tf.keras.layers.Dense(output_shape_dense, input_shape=(input_shape_dense,),
                                               name='layer_dense', activation=self.activation)
        self.dense_out.build((input_shape_dense,))

        super(RNNAggregator, self).build(())

        return

    def call(self, self_nodes, neigh_nodes, len_adj_nodes, training=True):
        neigh_cell = self.cell(neigh_nodes)

        concat = tf.concat([self_nodes, neigh_cell], axis=1)

        output = self.dense_out(concat)
        output = self.activation(output)

        return output