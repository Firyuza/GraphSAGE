import numpy as np
import tensorflow as tf
import scipy.io

from tqdm import tqdm

from .custom import CustomDataset
from .registry import DATASETS

@DATASETS.register_module
class ProteinDataset(CustomDataset):

    def load_annotations(self, ann_file):
        mat = scipy.io.loadmat(ann_file)  # 'datasets/%s.mat' % dataset_name)

        input = mat[self.dataset_name]
        labels = mat['l' +self. dataset_name.lower()]
        labels = labels - min(labels)
        labels = list(np.asarray(np.reshape(labels, [-1]), dtype=np.int32))

        node_labels = input['nl']
        v_labels = 0
        for i in range(node_labels.shape[1]):
            v_labels = max(v_labels, max(node_labels[0, i]['values'][0, 0])[0])

        e_labels = 1
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
            # V = np.ones([no_nodes, v_labels])
            # for l in range(v_labels):
            #     V[..., l] = np.equal(node_labels[0, i]['values'][0, 0][..., 0], l + 1).astype(np.float32)
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
        diff = self.data_infos['max_nrof_nodes'] - len(self.data_infos['Nodes'][graph_id])
        if diff > 0:
            # vertices = np.concatenate([self.data_infos['Nodes'][graph_id],
            #                            np.zeros((diff, 3))], axis=0)
            vertices = np.concatenate([self.data_infos['Nodes'][graph_id],
                                       np.zeros((diff), dtype=np.int32)], axis=0)
            right = np.concatenate([self.data_infos['Adj_list'][graph_id],
                                    np.zeros((len(self.data_infos['Adj_list'][graph_id]), diff))], axis=1)
            full_adj_list = np.concatenate([right, np.zeros((diff, diff + len(self.data_infos['Adj_list'][graph_id])))], axis=0)
        else:
            vertices = self.data_infos['Nodes'][graph_id]
            full_adj_list = self.data_infos['Adj_list'][graph_id]

        return np.asarray(vertices, np.int32), np.asarray(full_adj_list, np.float32), \
               self.data_infos['graph_size'][graph_id], self.data_infos['Labels'][graph_id]

    def prepare_train_data(self, graph_id):
        return tf.py_function(self.__py_func_map,[graph_id],
                              [tf.int32, tf.float32, tf.int32, tf.int32])

    def prepare_test_data(self, graph_id):
        return