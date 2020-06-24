import tensorflow as tf

from ..registry import ATTENTION_LAYER
from ..builder import build_attention_mechanism

@ATTENTION_LAYER.register_module
class GATLayer(tf.keras.layers.Layer):
    def __init__(self, attention_mechanism, attention_heads, activation, output_activation):
        super(GATLayer, self).__init__()

        self.activation = getattr(tf.nn, activation)
        self.output_activation = getattr(tf.nn, output_activation)
        self.attention_heads = attention_heads

        self.attention_mechanisms = []
        for _ in range(self.attention_heads):
            self.attention_mechanisms.append(build_attention_mechanism(attention_mechanism))

        return

    def build(self, input_shape, shared_output_shape, output_shape=1):
        for i in range(self.attention_heads):
            self.attention_mechanisms[i].build(input_shape, shared_output_shape, output_shape)

        super(GATLayer, self).build(())

        return

    def call(self, self_nodes, neigh_nodes, training=True):
        multi_head_self_nodes = None
        for i in range(self.attention_heads):
            self_nodes_upd, neigh_nodes_upd, coefficients = self.attention_mechanisms[i](self_nodes, neigh_nodes, training)
            coefficients = self.activation(coefficients)
            coefficients = tf.nn.softmax(tf.squeeze(coefficients, axis=2))
            coefficients = tf.expand_dims(coefficients, axis=2)
            self_nodes_upd = self.output_activation(tf.reduce_sum(tf.multiply(coefficients, neigh_nodes_upd), axis=1))
            if multi_head_self_nodes is None:
                multi_head_self_nodes = self_nodes_upd
            else:
                multi_head_self_nodes = tf.concat([multi_head_self_nodes, self_nodes_upd], axis=1)

        return multi_head_self_nodes