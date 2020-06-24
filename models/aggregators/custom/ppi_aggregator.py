import numpy as np
import tensorflow as tf

from ..base_aggregator import BaseAggregator
from ...registry import CUSTOM_AGGREGATOR

@CUSTOM_AGGREGATOR.register_module
class PPIAggregator(BaseAggregator):
    def __init__(self, nrof_neigh_per_batch, depth, aggregators_shape, aggregator_type,
                 attention_shapes=None):
        super(PPIAggregator, self).__init__(depth, aggregators_shape, aggregator_type, attention_shapes)

        self.nrof_neigh_per_batch = nrof_neigh_per_batch

    def build(self, input_shape):

        super(PPIAggregator, self).build(input_shape)

    def call_train(self, embedded_graph_nodes, graph_sizes, labels,
                    all_rnd_indices, all_rnd_adj_mask, all_len_adj_nodes):
        nrof_graphs = len(graph_sizes)

        batch_self_nodes = None
        for k in range(self.depth):
            batch_graphs_nodes = None
            for g_i in range(nrof_graphs):
                embedded_chosen_graph_nodes = tf.multiply(tf.expand_dims(all_rnd_adj_mask[g_i][k][:graph_sizes[g_i], :],
                                                                         axis=2),
                                                          tf.gather(embedded_graph_nodes[g_i] if k == 0
                                                                    else tf.gather(updated_graph_nodes,
                                                                                   tf.range(sum(graph_sizes[:g_i]),
                                                                                            sum(graph_sizes[
                                                                                                : (g_i + 1)]))),
                                                                    all_rnd_indices[g_i][k][:graph_sizes[g_i], :]))
                if batch_graphs_nodes is None:
                    batch_graphs_nodes = embedded_chosen_graph_nodes
                    batch_self_nodes = embedded_graph_nodes[g_i][:graph_sizes[g_i]]
                else:
                    batch_graphs_nodes = tf.concat([batch_graphs_nodes, embedded_chosen_graph_nodes],
                                                   axis=0)
                    batch_self_nodes = tf.concat([batch_self_nodes, embedded_graph_nodes[g_i][:graph_sizes[g_i]]],
                                                 axis=0)

            len_adj_nodes_per_graph = []
            [len_adj_nodes_per_graph.extend(len_adj[:graph_sizes[len_i]])
             for len_i, len_adj in enumerate(all_len_adj_nodes[:, k, :])]
            updated_graph_nodes = self.aggregator_layers[k](batch_self_nodes if k == 0 else updated_graph_nodes,
                                                            batch_graphs_nodes, len_adj_nodes_per_graph, training=True)

        updated_graph_nodes = tf.math.l2_normalize(updated_graph_nodes, axis=1)

        return updated_graph_nodes

    def call_test(self, embedded_graph_nodes, graph_sizes, labels,
                    all_rnd_indices, all_rnd_adj_mask, all_len_adj_nodes):
        nrof_graphs = len(graph_sizes)

        batch_self_nodes = None
        for k in range(self.depth):
            batch_graphs_nodes = None
            for g_i in range(nrof_graphs):
                embedded_chosen_graph_nodes = tf.multiply(tf.expand_dims(all_rnd_adj_mask[g_i][k][:graph_sizes[g_i], :],
                                                                         axis=2),
                                                          tf.gather(embedded_graph_nodes[g_i] if k == 0
                                                                    else tf.gather(updated_graph_nodes,
                                                                                   tf.range(sum(graph_sizes[:g_i]),
                                                                                            sum(graph_sizes[
                                                                                                : (g_i + 1)]))),
                                                                    all_rnd_indices[g_i][k][:graph_sizes[g_i], :]))
                if batch_graphs_nodes is None:
                    batch_graphs_nodes = embedded_chosen_graph_nodes
                    batch_self_nodes = embedded_graph_nodes[g_i][:graph_sizes[g_i]]
                else:
                    batch_graphs_nodes = tf.concat([batch_graphs_nodes, embedded_chosen_graph_nodes],
                                                   axis=0)
                    batch_self_nodes = tf.concat([batch_self_nodes, embedded_graph_nodes[g_i][:graph_sizes[g_i]]],
                                                 axis=0)

            len_adj_nodes_per_graph = []
            [len_adj_nodes_per_graph.extend(len_adj[:graph_sizes[len_i]])
             for len_i, len_adj in enumerate(all_len_adj_nodes[:, k, :])]
            updated_graph_nodes = self.aggregator_layers[k](batch_self_nodes if k == 0 else updated_graph_nodes,
                                                            batch_graphs_nodes, len_adj_nodes_per_graph, training=False)

        updated_graph_nodes = tf.math.l2_normalize(updated_graph_nodes, axis=1)

        return updated_graph_nodes