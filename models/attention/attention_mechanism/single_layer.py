import tensorflow as tf

from ...registry import ATTENTION_MECHANISM

@ATTENTION_MECHANISM.register_module
class SingleLayerMechanism(tf.keras.layers.Layer):
    def __init__(self):
        super(SingleLayerMechanism, self).__init__()

        return

    def build(self, input_shape, output_shape=1):
        self.dense = tf.keras.layers.Dense(output_shape, input_shape=(input_shape,),
                                           name='layer_dense', activation=None)
        self.dense.build((input_shape,))

        super(SingleLayerMechanism, self).build(())

        return

    def call(self, self_nodes, neigh_nodes):
        concat = tf.concat([tf.stack([self_nodes] * tf.shape(neigh_nodes)[1].numpy(), axis=1), neigh_nodes], axis=2)
        output = self.dense(concat)

        return output

