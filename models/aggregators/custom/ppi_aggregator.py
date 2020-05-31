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
            [len_adj_nodes_per_graph.extend(len_adj[:graph_sizes[len_i]]) for len_i, len_adj in enumerate(all_len_adj_nodes[:, k, :])]
            updated_graph_nodes = self.aggregator_layers[k](batch_self_nodes if k == 0 else updated_graph_nodes,
                                                            batch_graphs_nodes, len_adj_nodes_per_graph)

        updated_graph_nodes = tf.math.l2_normalize(updated_graph_nodes, axis=1)

        return updated_graph_nodes

    def call_test(self, graphs_nodes, graphs_adj_list, graph_sizes, labels=None):
        nrof_graphs = len(graphs_adj_list)

        embedded_graph_nodes = tf.gather(self.node_embedding, graphs_nodes)
        embedded_graph_nodes = tf.nn.leaky_relu(embedded_graph_nodes)

        batch_self_nodes = None
        for k in range(self.depth):
            len_adj_nodes_per_graph = []
            batch_graphs_nodes = None

            for g_i in range(nrof_graphs):
                rnd_indices = []
                rnd_adj_mask = []
                len_adj_nodes = []
                for i_v in range(graph_sizes[g_i]):
                    rnd_adj_nodes_indices = np.random.choice(
                        np.arange(graph_sizes[g_i]),
                        self.nrof_neigh_per_batch)
                    rnd_adj_nodes = tf.gather(graphs_adj_list[g_i][i_v], rnd_adj_nodes_indices)
                    rnd_indices.append(rnd_adj_nodes_indices)

                    nrof_adj = len(np.where(rnd_adj_nodes != 0)[0])
                    len_adj_nodes.append(nrof_adj if nrof_adj > 0 else 3e-5)

                    rnd_adj_mask.append(rnd_adj_nodes)

                len_adj_nodes_per_graph.extend(len_adj_nodes)

                embedded_chosen_graph_nodes = tf.multiply(tf.expand_dims(rnd_adj_mask, axis=2),
                                                          tf.gather(embedded_graph_nodes[g_i] if k == 0
                                                                    else tf.gather(updated_graph_nodes,
                                                                                   tf.range(sum(graph_sizes[:g_i]),
                                                                                            sum(graph_sizes[
                                                                                                : (g_i + 1)]))),
                                                                    rnd_indices))
                if batch_graphs_nodes is None:
                    batch_graphs_nodes = embedded_chosen_graph_nodes
                    batch_self_nodes = embedded_graph_nodes[g_i][:graph_sizes[g_i]]
                else:
                    batch_graphs_nodes = tf.concat([batch_graphs_nodes, embedded_chosen_graph_nodes],
                                                   axis=0)
                    batch_self_nodes = tf.concat([batch_self_nodes, embedded_graph_nodes[g_i][:graph_sizes[g_i]]],
                                                 axis=0)

            if k == 0:
                updated_graph_nodes = self.aggregator_layers[k](batch_self_nodes,
                                                                batch_graphs_nodes, len_adj_nodes_per_graph, training=False)
            else:
                updated_graph_nodes = self.aggregator_layers[k](updated_graph_nodes,
                                                                batch_graphs_nodes, len_adj_nodes_per_graph, training=False)

        entire_batch_graphs = None
        for g_i in range(nrof_graphs):
            embedded_graph = tf.gather(updated_graph_nodes,
                                       tf.range(sum(graph_sizes[:g_i]),
                                                sum(graph_sizes[: (g_i + 1)])))
            embedded_graph = tf.expand_dims(tf.reduce_mean(embedded_graph, axis=0), 0)
            if entire_batch_graphs is None:
                entire_batch_graphs = embedded_graph
            else:
                entire_batch_graphs = tf.concat([entire_batch_graphs, embedded_graph], axis=0)

        return entire_batch_graphs