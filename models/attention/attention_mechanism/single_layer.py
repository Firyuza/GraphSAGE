import tensorflow as tf

from ...registry import ATTENTION_MECHANISM

@ATTENTION_MECHANISM.register_module
class SingleLayerMechanism(tf.keras.layers.Layer):
    def __init__(self):
        super(SingleLayerMechanism, self).__init__()

        return

    def build(self, input_shape, shared_output_shape, output_shape=1):
        self.shared_dense = tf.keras.layers.Dense(shared_output_shape, input_shape=(input_shape,),
                                           name='layer_shared_dense', activation=None)
        self.shared_dense.build((input_shape,))

        self.bn = tf.keras.layers.BatchNormalization()
        self.bn.build((None, shared_output_shape))

        self.dense = tf.keras.layers.Dense(output_shape, input_shape=(2 * shared_output_shape,),
                                           name='layer_dense', activation=None)
        self.dense.build((2 * shared_output_shape,))

        super(SingleLayerMechanism, self).build(())

        return

    def call(self, self_nodes, neigh_nodes, training=True):
        self_nodes = self.shared_dense(self_nodes)
        self_nodes = self.bn(self_nodes, training=training)
        neigh_nodes = self.shared_dense(neigh_nodes)
        neigh_nodes_upd = self.bn(tf.reshape(neigh_nodes, [-1, int(neigh_nodes.shape[-1])]), training=training)
        neigh_nodes = tf.reshape(neigh_nodes_upd, list(neigh_nodes.shape))

        concat = tf.concat([tf.stack([self_nodes] * tf.shape(neigh_nodes)[1].numpy(), axis=1), neigh_nodes], axis=2)
        output = self.dense(concat)

        return self_nodes, neigh_nodes, output

