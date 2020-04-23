import tensorflow as tf

from ..registry import AGGREGATOR

@AGGREGATOR.register_module
class MeanAggregator(tf.keras.layers.Layer):
    def __init__(self, activation):
        super(MeanAggregator, self).__init__()

        self.activation = getattr(tf.nn, activation)

    def build(self, input_shape, output_shape=1):
        self.dense = tf.keras.layers.Dense(output_shape, input_shape=(input_shape,),
                                           name='layer_dense', activation=None)
        self.dense.build((input_shape,))

        super(MeanAggregator, self).build(())

        return

    def call(self, self_nodes, neigh_nodes, len_adj_nodes, training=True):

        mean_neigh = tf.divide(tf.reduce_sum(neigh_nodes, axis=1), tf.expand_dims(len_adj_nodes, 1))

        concat = tf.concat([self_nodes, mean_neigh], axis=1)
        concat = self.activation(concat)

        output = self.dense(concat)

        return output