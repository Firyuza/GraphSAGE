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

        graphs_size = []
        for i_g, graph in enumerate(data['Graph_nodes']):
            graphs_size.append(len(graph))

        return {
            'G': data['G'],
            'nodes': data['nodes'],
            'feats': data['feats'],
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
        rnd_node_indices = np.random.permutation(len(self.data_infos['nodes']))

        return rnd_node_indices

    def __py_func_map(self, node_id):
        node_id = self.data_infos['nodes'][node_id]
        self_node_embd = self.data_infos['feats'][self.data_infos['id_map'][node_id]]

        assert not self.data_infos['G'][node_id]['test'] and not self.data_infos['G'][node_id]['val']

        neighbors = np.array([self.data_infos['id_map'][neighbor]
                              for neighbor in self.data_infos['G'].neighbors(node_id)
                              if (not self.data_infos['G'][neighbor]['train_removed'])])
        neighbors = np.setdiff1d(neighbors, self.data_infos['id_map'][node_id])
        diff = self.nrof_neigh_per_batch - len(neighbors)
        neigh = np.random.choice(neighbors, min(self.nrof_neigh_per_batch, len(neighbors)))
        nrof_real_adj = len(neigh)
        if diff > 0:
            neigh = np.concatenate([neigh, [neighbors[0]] * diff])
        neigh_embd = self.data_infos['feats'][neigh]

        label = self.data_infos['class_map'][node_id]

        return np.asarray(self_node_embd, np.float32), nrof_real_adj, label, \
               np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32), \
               np.asarray(neigh_embd, dtype=np.float32)

    def prepare_train_data(self, graph_id):
        return tf.py_function(self.__py_func_map, [graph_id],
                              [tf.float32, tf.int32, tf.int32,
                               tf.int32, tf.float32, tf.float32])

    def prepare_test_data(self, graph_id):
        return tf.py_function(self.__py_func_map, [graph_id],
                              [tf.float32, tf.int32, tf.int32,
                               tf.int32, tf.float32, tf.float32])