import numpy as np
import tensorflow as tf

from sklearn.metrics import f1_score

from ..registry import GRAPH
from ..builder import build_custom_aggregator, build_loss, build_accuracy
from .base import BaseGraph

@GRAPH.register_module
class GraphSAGE(BaseGraph):
    def __init__(self, in_shape, out_shape, activation,
                 custom_aggregator,
                 loss_cls, accuracy_cls,
                 train_cfg, test_cfg):
        super(GraphSAGE, self).__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape

        self.activation = activation
        if self.activation is not None:
            self.activation = getattr(tf.nn, activation)

        self.custom_aggregator = build_custom_aggregator(custom_aggregator)

        self.loss_graph = build_loss(loss_cls)

        self.accuracy_cls = accuracy_cls
        self.accuracy = build_accuracy(accuracy_cls) #f1_score

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def reset_model_states(self):
        self.accuracy.reset_states()

        return

    def build_model(self, input_shape=()):
        self.output_dense = tf.keras.layers.Dense(self.out_shape, input_shape=(self.in_shape,),
                                           name='dense_output', activation=None)
        self.output_dense.build((self.in_shape,))

        self.custom_aggregator.build(input_shape)

        return

    def call_accuracy(self, predictions, labels):
        acc_value = self.accuracy(labels, predictions)
            # self.accuracy(labels.numpy(), np.round(predictions.numpy()), average='micro')
        # if len(acc_value.shape) > 0:
        #     acc_value = tf.reduce_mean(acc_value)
        return {self.accuracy_cls.type: acc_value}

    def call_loss(self, graph_nodes, labels):
        losses = {}
        loss = self.loss_graph(labels, graph_nodes)

        total_loss = loss

        if self.train_cfg.reg_loss is not None:
            reg_loss_type = getattr(tf.nn, self.train_cfg.reg_loss.type)
            reg_loss = self.train_cfg.reg_loss.weight_decay *\
                          tf.add_n([reg_loss_type(w) for w in self.trainable_variables])
            # self.add_loss(l2_reg_loss)
            total_loss += reg_loss

            losses.update({'reg_loss': reg_loss})

        losses.update({'total_loss': total_loss,
                       'loss': loss})

        return losses

    def call_train(self, graphs_nodes, graph_sizes, labels,
                   all_rnd_indices, all_rnd_adj_mask, all_len_adj_nodes):
        results = {}

        updated_graph_nodes = self.custom_aggregator(graphs_nodes, graph_sizes, labels,
                                                     all_rnd_indices, all_rnd_adj_mask, all_len_adj_nodes)
        updated_graph_nodes = self.output_dense(updated_graph_nodes)

        if len(labels.shape) > 1:
            batch_labels = None
            for g_i in range(len(graphs_nodes)):
                graph_labels = tf.gather(labels[g_i],
                                         tf.range(sum(graph_sizes[:g_i]),
                                                  sum(graph_sizes[: (g_i + 1)])))
                if batch_labels is None:
                    batch_labels = graph_labels
                else:
                    batch_labels = tf.concat([batch_labels, graph_labels], axis=0)
        else:
            batch_labels = np.expand_dims(labels, 1)

        predictions = self.activation(updated_graph_nodes)
        results['output'] = predictions
        results.update(self.call_loss(predictions, batch_labels))
        results.update(self.call_accuracy(predictions, batch_labels))

        return results

    def call_test(self, graphs_nodes, graphs_adj_list, graph_sizes, labels=None):
        results = {}

        updated_graph_nodes = self.custom_aggregator(graphs_nodes, graphs_adj_list, graph_sizes, labels)
        updated_graph_nodes = self.output_dense(updated_graph_nodes)

        predictions = self.activation(updated_graph_nodes)
        results['output'] = predictions
        if labels is not None:
            results.update(self.call_loss(updated_graph_nodes, labels))
            results.update(self.call_accuracy(predictions, labels))

        return results