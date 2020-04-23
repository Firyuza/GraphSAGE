import numpy as np
import scipy.io
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from itertools import chain
from collections import defaultdict

def load_protein_dataset(path, dataset_name):
    # if dataset_name not in chemical_datasets_list:
    #    print_ext('Dataset doesn\'t exist. Options:', chemical_datasets_list)
    #    return
    mat = scipy.io.loadmat(path) # 'datasets/%s.mat' % dataset_name)

    input = mat[dataset_name]
    labels = mat['l' + dataset_name.lower()]
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
        V = np.ones([no_nodes, v_labels])
        for l in range(v_labels):
            V[..., l] = np.equal(node_labels[0, i]['values'][0, 0][..., 0], l + 1).astype(np.float32)
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

    return samples_V, \
           samples_A, \
           labels, \
           max_no_nodes, graph_sizes, min_adj_nodes