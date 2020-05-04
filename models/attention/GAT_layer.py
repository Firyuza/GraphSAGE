import tensorflow as tf

from ..registry import ATTENTION_LAYER
from ..builder import build_attention_mechanism

@ATTENTION_LAYER.register_module
class GATLayer(tf.keras.layers.Layer):
    def __init__(self, attention_mechanism, activation):
        super(GATLayer, self).__init__()

        self.activation = getattr(tf.nn, activation)

        self.attention_mechanism = build_attention_mechanism(attention_mechanism)

        return

    def build(self, input_shape, output_shape=1):
        self.attention_mechanism.build(input_shape, output_shape)

        super(GATLayer, self).build(())

        return

    def call(self, self_nodes, neigh_nodes):
        output = self.attention_mechanism(self_nodes, neigh_nodes)
        output = self.activation(output)
        output = tf.nn.softmax(tf.squeeze(output, axis=2))
        output = tf.expand_dims(output, axis=2)

        return output