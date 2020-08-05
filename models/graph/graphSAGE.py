import numpy as np
import tensorflow as tf

from sklearn.metrics import f1_score

from ..registry import GRAPH
from ..builder import build_aggregator_layers, build_loss, build_accuracy
from .base import BaseGraph

@GRAPH.register_module
class GraphSAGE(BaseGraph):
    def __init__(self, in_shape, out_shape, activation,
                 aggregator_layers,
                 loss_cls, accuracy_cls,
                 train_cfg, test_cfg):
        super(GraphSAGE, self).__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape

        self.activation = activation
        if self.activation is not None:
            self.activation = getattr(tf.nn, activation)

        self.aggregator_layers = build_aggregator_layers(aggregator_layers)

        self.loss_graph = build_loss(loss_cls)

        self.accuracy_cls = accuracy_cls
        self.accuracy = build_accuracy(accuracy_cls)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def reset_model_states(self):
        self.accuracy.reset_states()

        return

    def build_model(self, input_shape=()):
        self.output_dense = tf.keras.layers.Dense(self.out_shape, input_shape=(self.in_shape,),
                                           name='dense_output', activation=None)
        self.output_dense.build((self.in_shape,))

        self.aggregator_layers.build(input_shape)

        return


    def call_accuracy(self, predictions, labels):
        def tf_f1_score(y_true, y_pred):
            """Computes 3 different f1 scores, micro macro
            weighted.
            micro: f1 score accross the classes, as 1
            macro: mean of f1 scores per class
            weighted: weighted average of f1 scores per class,
                    weighted from the support of each class


            Args:
                y_true (Tensor): labels, with shape (batch, num_classes)
                y_pred (Tensor): model's predictions, same shape as y_true

            Returns:
                tuple(Tensor): (micro, macro, weighted)
                            tuple of the computed f1 scores
            """

            f1s = [0, 0, 0]
            precisions = [0, 0]
            recalls = [0, 0]

            y_true = tf.cast(y_true, tf.float64)
            y_pred = tf.cast(tf.round(y_pred), tf.float64)

            for i, axis in enumerate([None, 0]):
                TP = tf.math.count_nonzero(y_pred * y_true, axis=axis)
                FP = tf.math.count_nonzero(y_pred * (y_true - 1), axis=axis)
                FN = tf.math.count_nonzero((y_pred - 1) * y_true, axis=axis)

                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = 2 * precision * recall / (precision + recall)

                precisions[i] = precision
                recalls[i] = recall
                f1s[i] = tf.reduce_mean(f1)

            weights = tf.reduce_sum(y_true, axis=0)
            weights /= tf.reduce_sum(weights)

            f1s[2] = tf.reduce_sum(f1 * weights)

            micro, macro, weighted = f1s

            return micro, macro, weighted, precisions, recalls

        micro, macro, weighted, precisions, recalls = tf_f1_score(labels, predictions)

        acc_value = self.accuracy(labels, predictions)
        output = {self.accuracy_cls.type: acc_value,
                'precision': precisions[0], 'recall': recalls[0]}

        # output = {**output,
        #           **{'precision_%d' % i: precisions[1][i] for i in range(len(precisions[1]))},
        #           **{'recall_%d' % i: recalls[1][i] for i in range(len(recalls[1]))}}

        return output

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

    def call_train(self, graphs_nodes, graph_sizes,
                   all_rnd_indices, all_rnd_adj_mask, all_len_adj_nodes, labels):
        results = {}

        updated_graph_nodes = self.aggregator_layers(graphs_nodes, graph_sizes,
                                                     all_rnd_indices, all_rnd_adj_mask, all_len_adj_nodes, labels,
                                                     train_mode=True)
        vis_embeddings = updated_graph_nodes
        updated_graph_nodes = self.output_dense(updated_graph_nodes)

        predictions = self.activation(updated_graph_nodes)
        results['output'] = predictions

        if len(labels.shape) > 1:
            batch_labels = None
            for g_i in range(len(graphs_nodes)):
                graph_labels = labels[g_i][:graph_sizes[g_i], :]
                if batch_labels is None:
                    batch_labels = graph_labels
                else:
                    batch_labels = tf.concat([batch_labels, graph_labels], axis=0)
        else:
            batch_labels = np.expand_dims(labels, 1)

        results.update(self.call_loss(predictions, batch_labels))
        results.update(self.call_accuracy(predictions, batch_labels))

        return results, vis_embeddings, batch_labels, predictions

    def call_test(self, graphs_nodes, graph_sizes,
                   all_rnd_indices, all_rnd_adj_mask, all_len_adj_nodes, labels=None):
        results = {}

        updated_graph_nodes = self.aggregator_layers(graphs_nodes, graph_sizes,
                                                     all_rnd_indices, all_rnd_adj_mask, all_len_adj_nodes, labels,
                                                     train_mode=False)
        vis_embeddings = updated_graph_nodes

        updated_graph_nodes = self.output_dense(updated_graph_nodes)

        predictions = self.activation(updated_graph_nodes)
        results['output'] = predictions

        if labels is not None:
            if len(labels.shape) > 1:
                batch_labels = None
                for g_i in range(len(graphs_nodes)):
                    graph_labels = labels[g_i][:graph_sizes[g_i], :]
                    if batch_labels is None:
                        batch_labels = graph_labels
                    else:
                        batch_labels = tf.concat([batch_labels, graph_labels], axis=0)
            else:
                batch_labels = np.expand_dims(labels, 1)

        results.update(self.call_loss(updated_graph_nodes, batch_labels))
        results.update(self.call_accuracy(predictions, batch_labels))

        return results, vis_embeddings, batch_labels, predictions