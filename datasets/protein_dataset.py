import numpy as np
import tensorflow as tf
import scipy.io

from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

from numba import njit, prange, jit

from .custom import CustomDataset
from .registry import DATASETS

@DATASETS.register_module
class ProteinDataset(CustomDataset):

    def load_annotations(self, ann_file):
        mat = scipy.io.loadmat(ann_file)

        input = mat[self.dataset_name]
        labels = mat['l' +self. dataset_name.lower()]
        labels = labels - min(labels)
        labels = list(np.asarray(np.reshape(labels, [-1]), dtype=np.int32))

        node_labels = input['nl']
        v_labels = 0
        for i in range(node_labels.shape[1]):
            v_labels = max(v_labels, max(node_labels[0, i]['values'][0, 0])[0])

        # For each sample
        samples_V = []
        samples_A = []
        max_no_nodes = 0
        graph_sizes = []
        min_adj_nodes = np.inf
        zero_adj_graphs = []
        for i in range(input.shape[1]):
            no_nodes = node_labels[0, i]['values'][0, 0].shape[0]
            max_no_nodes = max(max_no_nodes, no_nodes)
            graph_sizes.append(no_nodes)
            V = node_labels[0, i]['values'][0, 0][..., 0] - 1
            samples_V.append(V)
            A = np.zeros([no_nodes, no_nodes])
            for j in range(no_nodes):
                if input[0, i]['al'][j, 0].shape[1] == 0:
                    zero_adj_graphs.append(i)
                else:
                    min_adj_nodes = min(min_adj_nodes, input[0, i]['al'][j, 0].shape[1])

                for k in range(input[0, i]['al'][j, 0].shape[1]):
                    A[j, input[0, i]['al'][j, 0][0, k] - 1] = 1

            samples_A.append(A)

        for zero_idx in zero_adj_graphs:
            samples_V.pop(zero_idx)
            samples_A.pop(zero_idx)
            labels.pop(zero_idx)
            graph_sizes.pop(zero_idx)

        return {'Nodes': samples_V,
               'Adj_list': samples_A,
               'Labels': labels,
               'max_nrof_nodes': max_no_nodes,
               'graph_size': graph_sizes,
               'nrof_min_adj_nodes': min_adj_nodes}

    def prepare_train_indices(self):
        rnd_graph_indices = np.random.permutation(len(self.data_infos['Nodes']))

        return rnd_graph_indices

    def __py_func_map(self, graph_id):

        def generate_neighbours(graphs_adj_list, graph_size, depth, nrof_neigh_per_batch):
            all_rnd_indices = []
            all_rnd_adj_mask = []
            all_len_adj_nodes = []

            for _ in range(depth):
                rnd_indices = []
                rnd_adj_mask = []
                len_adj_nodes = []
                for i_v in range(graph_size):
                    rnd_adj_nodes_indices = np.random.choice(
                        np.arange(graph_size),
                        nrof_neigh_per_batch)
                    rnd_adj_nodes = graphs_adj_list[i_v][rnd_adj_nodes_indices]
                    rnd_indices.append(rnd_adj_nodes_indices)

                    nrof_adj = len(np.where(rnd_adj_nodes != 0)[0])
                    len_adj_nodes.append(nrof_adj if nrof_adj > 0 else 3e-5)

                    rnd_adj_mask.append(rnd_adj_nodes)

                all_rnd_indices.append(rnd_indices)
                all_rnd_adj_mask.append(rnd_adj_mask)
                all_len_adj_nodes.append(len_adj_nodes)

            return all_rnd_indices, all_rnd_adj_mask, all_len_adj_nodes

        graph_size = self.data_infos['graph_size'][graph_id]
        graphs_adj_list = self.data_infos['Adj_list'][graph_id]

        diff = self.data_infos['max_nrof_nodes'] - len(self.data_infos['Nodes'][graph_id])
        if diff > 0:
            vertices = np.concatenate([self.data_infos['Nodes'][graph_id],
                                       np.zeros((diff), dtype=np.int32)], axis=0)
        else:
            vertices = self.data_infos['Nodes'][graph_id]

        all_rnd_indices, all_rnd_adj_mask, all_len_adj_nodes = generate_neighbours(graphs_adj_list, graph_size,
                                                                                   self.depth, self.nrof_neigh_per_batch)
        for i in range(self.depth):
            if diff > 0:
                all_rnd_indices[i] = np.concatenate([all_rnd_indices[i],
                                              np.zeros((diff, self.nrof_neigh_per_batch), dtype=np.int32)], axis=0)
                all_rnd_adj_mask[i] = np.concatenate([all_rnd_adj_mask[i],
                                               np.zeros((diff, self.nrof_neigh_per_batch), dtype=np.int32)],
                                              axis=0)
                all_len_adj_nodes[i] = np.concatenate([all_len_adj_nodes[i],
                                                np.zeros((diff), dtype=np.int32)], axis=0)

        return np.asarray(vertices, np.int32), self.data_infos['graph_size'][graph_id], \
               self.data_infos['Labels'][graph_id], \
               np.asarray(all_rnd_indices, dtype=np.int32), np.asarray(all_rnd_adj_mask, dtype=np.float32), \
               np.asarray(all_len_adj_nodes, dtype=np.float32)

    def prepare_train_data(self, graph_id):
        return tf.py_function(self.__py_func_map,[graph_id],
                              [tf.int32, tf.int32, tf.int32,
                               tf.int32, tf.float32, tf.float32])

    def prepare_test_data(self, graph_id):
        return