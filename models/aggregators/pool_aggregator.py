import tensorflow as tf

from ..registry import AGGREGATOR
from ..builder import build_attention_layer

@AGGREGATOR.register_module
class PoolAggregator(tf.keras.layers.Layer):
    def __init__(self, activation, pool_op, attention_layer=None):
        super(PoolAggregator, self).__init__()

        self.activation = getattr(tf.nn, activation)
        self.pool_op = pool_op

        self.attention_layer = attention_layer
        if attention_layer is not None:
            self.attention_layer = build_attention_layer(attention_layer)

    def build(self, input_shape, transform_output_shape, dense_input_shape, output_shape=1,
              attention_in_shape=None, attention_shared_out_shape=None, attention_out_shape=None):
        self.transform_node_weight = tf.keras.layers.Dense(transform_output_shape, input_shape=(input_shape,),
                                           name='transform_node_weight', activation=None)
        self.transform_node_weight.build((input_shape, ))

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn1.build((None, transform_output_shape))

        self.dense_out = tf.keras.layers.Dense(output_shape, input_shape=(dense_input_shape,),
                                           name='layer_dense', activation=None)
        self.dense_out.build((dense_input_shape,))
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn2.build((None, output_shape))

        if self.attention_layer is not None:
            self.attention_layer.build(attention_in_shape, attention_shared_out_shape, attention_out_shape)

        super(PoolAggregator, self).build(())

        return

    def call(self, self_nodes, neigh_nodes, len_adj_nodes, training=True):
        if self.attention_layer is not None:
            self_nodes = self.attention_layer(self_nodes, neigh_nodes, training)

        neigh = self.transform_node_weight(neigh_nodes)
        neigh = self.bn1(neigh, training=training)
        neigh = getattr(tf, self.pool_op)(neigh, axis=1)

        concat = tf.concat([self_nodes, neigh], axis=1)
        concat = self.activation(concat)

        output = self.dense_out(concat)
        output = self.bn2(output, training=training)
        output = self.activation(output)

        return output