import torch
import numpy as np
import networkx as nx
from numpy import linalg as LA
from sklearn.preprocessing import OneHotEncoder
import argparse

def LDP(g, key='deg'):
    x = np.zeros([len(g.nodes()), 5])

    deg_dict = dict(nx.degree(g))
    for n in g.nodes():
        g.nodes[n][key] = deg_dict[n]

    for i in g.nodes():
        nodes = g[i].keys()

        nbrs_deg = [g.nodes[j][key] for j in nodes]

        if len(nbrs_deg) != 0:
            x[i] = [
                np.mean(nbrs_deg),
                np.min(nbrs_deg),
                np.max(nbrs_deg),
                np.std(nbrs_deg),
                np.sum(nbrs_deg)
            ]

    return x

def binning(a, n_bins=10):
    n_graphs = a.shape[0]
    n_nodes = a.shape[1]
    _, bins = np.histogram(a, n_bins)
    binned = np.digitize(a, bins)
    binned = binned.reshape(-1, 1)
    enc = OneHotEncoder()
    return enc.fit_transform(binned).toarray().reshape(n_graphs, n_nodes, -1).astype(np.float32)


def compute_x(a1, node_features):
    # construct node features X
    if node_features == 'identity':
        x = torch.cat([torch.diag(torch.ones(a1.shape[1]))] * a1.shape[0]).reshape([a1.shape[0], a1.shape[1], -1])
        x1 = x.clone()

    # elif args.node_features == 'node2vec':
    #     X = np.load(f'./{args.dataset_name}_{args.modality}.emb', allow_pickle=True).astype(np.float32)
    #     x1 = torch.from_numpy(X)

    elif node_features == 'degree':
        a1b = (a1 != 0).float()
        x1 = a1b.sum(dim=2, keepdim=True)

    elif node_features == 'degree_bin':
        a1b = (a1 != 0).float()
        x1 = binning(a1b.sum(dim=2))

    elif node_features == 'adj': # edge profile
        x1 = a1.float()

    elif node_features == 'LDP': # degree profile
        a1b = (a1 != 0).float()
        x1 = []
        n_graphs: int = a1.shape[0]
        for i in range(n_graphs):
            x1.append(LDP(nx.from_numpy_array(a1b[i].numpy())))

    elif node_features == 'eigen':
        _, x1 = LA.eig(a1.numpy())

    x1 = torch.Tensor(x1).float()
    return x1

if __name__ == '__main__':
    a = torch.rand(3,5,5)
    a1 = a.numpy()
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_features", "-f", default='identity')
    parser.add_argument("--dataset_name")
    parser.add_argument("--modality")
    args = parser.parse_args()
    # out = compute_x_np(a1, {'node_features': args.node_features})
    out = compute_x(a, args)
    print(out.shape)
    print(out)
    # print(out1)