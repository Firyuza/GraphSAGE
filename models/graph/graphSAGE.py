import numpy as np
import tensorflow as tf

from ..registry import GRAPH
from ..builder import build_custom_aggregator, build_loss
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
        self.activation = getattr(tf.nn, activation)

        self.custom_aggregator = build_custom_aggregator(custom_aggregator)

        self.loss_graph = build_loss(loss_cls)

        self.accuracy_cls = accuracy_cls
        self.accuracy = getattr(tf.keras.metrics, accuracy_cls.type)()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def reset_model_states(self):
        self.accuracy.reset_states()

        return

    def build_model(self, input_shape=()):
        self.output_dense = tf.keras.layers.Dense(self.out_shape, input_shape=(self.in_shape,),
                                           name='dense_output', activation=self.activation)
        self.output_dense.build((self.in_shape,))

        self.custom_aggregator.build(input_shape)

        return

    def call_accuracy(self, predictions, labels):
        acc_value = self.accuracy(labels, predictions)

        return {self.accuracy_cls.type: acc_value}

    def call_loss(self, entire_batch_graphs, labels):
        losses = {}
        loss = self.loss_graph(labels, entire_batch_graphs)

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

    def call_train(self, graphs_nodes, graphs_adj_list, graph_sizes, labels):
        results = {}

        entire_batch_graphs = self.custom_aggregator(graphs_nodes, graphs_adj_list, graph_sizes, labels)
        entire_batch_graphs = self.output_dense(entire_batch_graphs)

        results['output'] = entire_batch_graphs
        results.update(self.call_loss(entire_batch_graphs, labels))
        results.update(self.call_accuracy(entire_batch_graphs, labels))

        return results

    def call_test(self, graphs_nodes, graphs_adj_list, graph_sizes, labels=None):
        results = {}

        entire_batch_graphs = self.custom_aggregator(graphs_nodes, graphs_adj_list, graph_sizes, labels)
        entire_batch_graphs = self.output_dense(entire_batch_graphs)

        results['output'] = entire_batch_graphs
        if labels is not None:
            results.update(self.call_loss(entire_batch_graphs, labels))
            results.update(self.call_accuracy(entire_batch_graphs, labels))

        return results