import tensorflow as tf
import numpy as np

from models_ import *
from datasets_ import *

samples_V, sample_A, labels, max_nof_nodes, graph_sizes, min_adj_nodes = \
    load_protein_dataset('/home/firiuza/MachineLearning/proteins.mat', 'proteins')

def map_func(graph_id):
    def py_map(graph_id):
        diff = max_nof_nodes - len(samples_V[graph_id])
        if diff > 0:
            vertices = np.concatenate([samples_V[graph_id], np.zeros((diff, 3))], axis=0)
            right = np.concatenate([sample_A[graph_id], np.zeros((len(sample_A[graph_id]), diff))], axis=1)
            full = np.concatenate([right, np.zeros((diff, diff + len(sample_A[graph_id])))], axis=0)
        else:
            vertices = samples_V[graph_id]
            full = sample_A[graph_id]

        return np.asarray(vertices, np.float32), np.asarray(full, np.float32), \
               labels[graph_id], graph_sizes[graph_id]

    return tf.py_function(py_map, [graph_id], [tf.float32, tf.float32, tf.int32, tf.int32])


graphSAGE = GraphSAGE(embd_shape=3, aggregator_type='MeanAggregator', depth=2, weight_decay=0.0005)
graphSAGE.build()
graphSAGE.summary()

BC_loss = tf.keras.losses.BinaryCrossentropy()
all_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='train_all_accuracy')

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            3e-3,
            decay_steps=1000,
            decay_rate=0.99,
            staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)


for i in range(5):
    print('EPOCH %d' % i)
    rnd_graph_indices = np.random.permutation(len(samples_V))
    dataset = tf.data.Dataset.from_tensor_slices((rnd_graph_indices)).map(map_func).batch(batch_size=16)

    for V, A, l, g_size in dataset:
        with tf.GradientTape() as tape:
            output = graphSAGE(V, A, g_size, len(A), nrof_neigh_per_batch=10)

            loss = BC_loss(l, output)
            reg_loss = sum(graphSAGE.losses)
            # total_loss = loss + reg_loss

        grads = tape.gradient(loss, graphSAGE.trainable_weights)
        optimizer.apply_gradients(zip(grads, graphSAGE.trainable_weights))

        all_accuracy_metric(l, output)
        print('BC loss: %.3f \t Reg Loss: %.3f \t Accuracy: %.3f' %
              (loss, reg_loss,  all_accuracy_metric.result()))

