import numpy as np
import tensorflow as tf
import pickle
import random
import json
import sys
import os

from tqdm import tqdm
from networkx.readwrite import json_graph
from joblib import Parallel, delayed

from .custom import CustomDataset
from .registry import DATASETS

@DATASETS.register_module
class PPIDataset(CustomDataset):

    def load_annotations(self, ann_file):
        with open(ann_file, 'rb') as f:
            data = pickle.load(f)

        valid_indices = []
        for i in range(len(data['Graph_nodes'])):
            if len(data['Graph_nodes'][i]) > 2:
                valid_indices.append(i)
        print('Nrof graphs: %d' % len(valid_indices))

        feats = [[]for _ in range(len(data['Graph_nodes']))]
        graphs_size = []
        for i_g, graph in enumerate(data['Graph_nodes']):
            graphs_size.append(len(graph))
            for nodeid in graph:
                feats[i_g].append(data['feats'][nodeid])

        return {
            'G': data['G'],
            'feats': feats,
            'id_map': data['id_map'],
            'class_map': data['class_map'],
            'Graph_nodes': data['Graph_nodes'],
            'Graph_feats': data['Graph_feats'],
            'Graph_adj_list': data['Graph_adj_list'],
            'Graph_labels': data['Graph_labels'],
            'Graph_degree': data['Graph_degree'],
            'max_nrof_nodes': data['max_nrof_nodes'],
            'valid_indices': valid_indices,
            'Graph_size': graphs_size
        }

    def prepare_train_indices(self):
        rnd_graph_indices = np.random.permutation(len(self.data_infos['valid_indices']))

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
                    exactly_adj_nodes = np.where(graphs_adj_list[i_v] != 0)[0]
                    non_adj_nodes = np.setdiff1d(np.arange(graph_size), exactly_adj_nodes)

                    rnd_adj_nodes_indices = np.unique(np.random.choice(
                        exactly_adj_nodes,
                        min(len(exactly_adj_nodes), nrof_neigh_per_batch)))
                    nrof_adj = len(rnd_adj_nodes_indices)
                    if nrof_adj < nrof_neigh_per_batch:
                        rnd_adj_nodes_indices = np.concatenate([rnd_adj_nodes_indices,
                                                                [non_adj_nodes[0]] *
                                                                (nrof_neigh_per_batch - nrof_adj)])
                    rnd_adj_nodes = graphs_adj_list[i_v][rnd_adj_nodes_indices]
                    rnd_indices.append(rnd_adj_nodes_indices)

                    len_adj_nodes.append(nrof_adj if nrof_adj > 0 else 3e-5)
                    rnd_adj_mask.append(rnd_adj_nodes)

                # TODO doesn't work fast
                # def generate_exact_adj_nodes(i_v):
                #     exactly_adj_nodes = np.where(graphs_adj_list[i_v] != 0)[0]
                #     non_adj_nodes = np.setdiff1d(np.arange(graph_size), exactly_adj_nodes)
                #
                #     rnd_adj_nodes_indices = np.unique(np.random.choice(
                #         exactly_adj_nodes,
                #         min(len(exactly_adj_nodes), nrof_neigh_per_batch)))
                #     nrof_adj = len(rnd_adj_nodes_indices)
                #     if nrof_adj < nrof_neigh_per_batch:
                #         rnd_adj_nodes_indices = np.concatenate([rnd_adj_nodes_indices,
                #                                                 [non_adj_nodes[0]] *
                #                                                 (nrof_neigh_per_batch - nrof_adj)])
                #     rnd_adj_nodes = graphs_adj_list[i_v][rnd_adj_nodes_indices]
                #
                #     return i_v, rnd_adj_nodes_indices, nrof_adj if nrof_adj > 0 else 3e-5, rnd_adj_nodes

                # with Parallel(n_jobs=-1, verbose=0) as parallel:
                #     result = parallel(delayed(generate_exact_adj_nodes)(idx) for idx in range(graph_size))
                #
                # for res in result:
                #     rnd_indices[res[0]] = res[1]
                #     len_adj_nodes[res[0]] = res[2]
                #     rnd_adj_mask[res[0]] = res[1]


                all_rnd_indices.append(rnd_indices)
                all_rnd_adj_mask.append(rnd_adj_mask)
                all_len_adj_nodes.append(len_adj_nodes)

            return all_rnd_indices, all_rnd_adj_mask, all_len_adj_nodes

        graph_id = self.data_infos['valid_indices'][graph_id]

        graph_size = self.data_infos['Graph_size'][graph_id]
        graphs_adj_list = self.data_infos['Graph_adj_list'][graph_id]

        diff = self.data_infos['max_nrof_nodes'] - len(self.data_infos['Graph_nodes'][graph_id])
        if diff > 0:
            vertices = np.concatenate([self.data_infos['feats'][graph_id],
                                       np.zeros((diff, len(self.data_infos['feats'][graph_id][0])), dtype=np.float32)],
                                      axis=0)
            labels = np.concatenate([self.data_infos['Graph_labels'][graph_id],
                                     np.zeros((diff, len(self.data_infos['Graph_labels'][graph_id][0])),
                                              dtype=np.int32)],
                                    axis=0)
        else:
            vertices = self.data_infos['feats'][graph_id]
            labels = self.data_infos['Graph_labels'][graph_id]

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

        return np.asarray(vertices, np.float32), graph_size, labels, \
               np.asarray(all_rnd_indices, dtype=np.int32), np.asarray(all_rnd_adj_mask, dtype=np.float32), \
               np.asarray(all_len_adj_nodes, dtype=np.float32)

    def prepare_train_data(self, graph_id):
        return tf.py_function(self.__py_func_map, [graph_id],
                              [tf.float32, tf.int32, tf.int32,
                               tf.int32, tf.float32, tf.float32])

    def prepare_test_data(self, graph_id):
        return