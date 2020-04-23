import tensorflow as tf

from ..registry import AGGREGATOR

@AGGREGATOR.register_module
class PoolAggregator(tf.keras.layers.Layer):
    def __init__(self, activation, pool_op):
        super(PoolAggregator, self).__init__()

        self.activation = getattr(tf.nn, activation)
        self.pool_op = pool_op

    def build(self, input_shape, transform_output_shape, dense_input_shape, output_shape=1):
        self.transform_node_weight = tf.keras.layers.Dense(transform_output_shape, input_shape=(input_shape,),
                                           name='transform_node_weight', activation=None)
        self.transform_node_weight.build((input_shape, ))

        self.dense_out = tf.keras.layers.Dense(output_shape, input_shape=(dense_input_shape,),
                                           name='layer_dense', activation=self.activation)
        self.dense_out.build((dense_input_shape,))

        super(PoolAggregator, self).build(())

        return

    def call(self, self_nodes, neigh_nodes, len_adj_nodes, training=True):
        neigh = self.transform_node_weight(neigh_nodes)
        neigh = getattr(tf, self.pool_op)(neigh, axis=1)

        concat = tf.concat([self_nodes, neigh], axis=1)
        concat = self.activation(concat)

        output = self.dense_out(concat)

        return output