
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.utils import degree
import torch
from scipy.sparse import coo_matrix
from scipy.io import mmread, mmwrite
import os
import magicgraph
from graph_coarsening import DoubleWeightedDiGraph,coarsening
import pickle as pkl
import numpy as np


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

def load_dataset(path, dataset_name="REDDIT-MULTI-5K"):
    graphs = TUDataset(path, dataset_name)
    if graphs.data.x is None:
        max_degree = 0
        degs = []
        for data in graphs:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 2000:
            graphs.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            graphs.transform = NormalizedDegree(mean, std)

    return graphs


class Data(object):
    def __init__(self,path,dataset,sfdp_path):
        self.path = path
        self.dataset = dataset
        self.sfdp_path = sfdp_path

    def load(self):
        def _make_adj(g, n_nodes):
            adj = np.zeros((n_nodes,n_nodes))
            edges, weights = g.get_int_edges()
            for k,(i,j) in enumerate(edges):
                adj[i,j] = weights[k]
            return adj
        def _enc_label(labels):
            labels = [l[0] for l in labels]
            # n_class = max(labels)+1
            # N = len(labels)
            # onehot = np.zeros((N, n_class))
            # onehot[range(N), labels] = 1.
            return np.array(labels)
        # def _padding_feat(feats):
        #     """
        #     feats: shape (bacth_size, K, N, D)
        #     """
        #     batch_size = len(feats)
        #     _, _, D = feats[0].shape
        #     max_node = max([feat.shape[1] for feat in feats])
        #     max_level = self.action_num
        #     padding_feats = np.zeros((batch_size, max_level, max_node, D))
        #     for i in range(batch_size):
        #         K,N,_ = feats[i].shape
        #         K = min(K, max_level)
        #         padding_feats[i,:K,:N,:] = feats[i][:K]
        #     return padding_feats
        # def _padding_adj(adjs):
        #     """
        #     adjs: shape (batch_size, K, N, N)
        #     """
        #     max_level = self.action_num
        #     batch_size = len(adjs)
        #     max_node = max([a.shape[1] for a in adjs])
        #     padding_adj = np.zeros((batch_size, max_level, max_node, max_node))
        #     for i in range(batch_size):
        #         K,N,_ = adjs[i].shape
        #         K = min(K, max_level)
        #         padding_adj[i,:K, :N, :N] = adjs[i][:K]
        #     return padding_adj
        
        def _serialize_graph(graphs,path):
            rows = []
            cols = []
            data = []
            offsets = []
            # big_offsets = []
            # levels = []
            # N = 0
            ofst = 0
            for i in range(len(graphs)):
                # edges = []
                # levels.append(len(graphs[i]))
                m = graphs[i][0].number_of_nodes()
                for j in range(len(graphs[i])):
                    di_egdes,weights = graphs[i][j].get_int_edges()
                    # rows,cols = [],[]
                    es = set()
                    cnt = 0
                    # ofst = m*j+N
                    for x,y in di_egdes:
                        if (y,x) not in es:
                            es.add((x,y))
                            rows.append(x+ofst)
                            cols.append(y+ofst)
                            cnt += 1
                            data.append(1)
                            # big_cnt += 1
                    ofst += m
                    offsets.append(cnt)
                # N += len(graphs[i]) * m
            coo = coo_matrix((data,(rows,cols)),shape=(ofst,ofst))
            mmwrite(path,coo)
            print('graph saved, shape', (ofst,ofst))
            return offsets

        def _deserialize_grpahs(path,offsets,node_nums,levels):
            coo = mmread(path)
            s = 0
            ofst = 0
            graphs = []
            ofstl = 0
            for i,l in enumerate(levels):
                m = node_nums[i]
                subgraphs = []
                ns = [0]+offsets[ofstl:ofstl+l]
                ns = np.cumsum(ns)
                for j in range(l):
                    graph = magicgraph.DiGraph()
                    # import pdb; pdb.set_trace()
                    rows = coo.row[s+ns[j]:s+ns[j+1]]
                    cols = coo.col[s+ns[j]:s+ns[j+1]]
                    # ofst = N+m*j
                    for x,y in zip(rows,cols):
                        x -= ofst
                        y -= ofst
                        graph[x].append(y)
                        graph[y].append(x)
                    ofst += m
                    subgraphs.append(DoubleWeightedDiGraph(graph))
                graphs.append(subgraphs)
                s += ns[-1]
                # N += m*l
                ofstl += l
            return graphs
        
        def _process_merged(merged_ls,graph_ls,feature_ls):
            mask_merged = []
            mask_node = []
            indices = range(len(merged_ls))
            for i in indices:
                level = len(merged_ls[i])
                edges,_ = graph_ls[i][0].get_int_edges()
                N = feature_ls[i].shape[0]
                mask_levels = np.zeros((level,N,N))
                mask1_levels = np.zeros((level,N))
                for j in range(level):
                    merged = merged_ls[i][j]
                    mask = mask_levels[j]
                    mask1 = mask1_levels[j]
                    for (a,b) in edges:
                        if a in merged and b in merged:
                            ma,mb = merged[a],merged[b]
                            if ma == mb:
                                if ma != a:
                                    mask[ma][a] = 1.
                                if mb != b:
                                    mask[mb][b] = 1.
                                mask[ma][ma] = 1.
                                mask1[ma] = 1.
                    edges, _ = graph_ls[i][j+1].get_int_edges()
                mask_merged.append(mask_levels)
                mask_node.append(mask1_levels)
            return mask_merged,mask_node

        if os.path.exists(self.path+'/cache'):
            print('load cache')
            with open(self.path+'/cache','rb') as f:
                cache = pkl.load(f)
                merged_bigls = cache['merged']
                offsets = cache['offsets']
                deg_ls = cache['avg_deg']
                levels = []
                for i in range(len(merged_bigls)):
                    levels.append(len(merged_bigls[i])+1)
                graphs = load_dataset(self.path,dataset_name=self.dataset)
                connected_indices = np.loadtxt(self.path+'/connected.txt',dtype=int)
                feat_ls = []
                label_ls = []
                node_nums = []
                print('load feat')
                for i in connected_indices:
                    feat_ls.append(graphs[i].x.numpy())
                    label_ls.append(graphs[i].y.numpy())
                    node_nums.append(graphs[i].x.shape[0])
                coarsened_graph_ls = _deserialize_grpahs(self.path+'/graphs.mtx',offsets,node_nums,levels)
                coarsened_graph_adjs = []
                print('make adj')
                for i in range(len(coarsened_graph_ls)):
                    adjs = []
                    for j in range(len(coarsened_graph_ls[i])):
                        adjs.append(_make_adj(coarsened_graph_ls[i][j],node_nums[i]))
                    coarsened_graph_adjs.append(np.array(adjs))
            print('cache loaded!!!')
        else:
            graphs = load_dataset(self.path,dataset_name=self.dataset)
            coarsened_graph_adjs = []
            feat_ls = []
            label_ls = []
            merged_bigls = []
            coarsened_graph_ls = []
            deg_ls = []
            # node_nums = []
            # levels = []
            not_connected = 0
            # fc = open(self.path+'/connected.txt','w')
            connected_indices = []
            for i,g in enumerate(graphs):
                print(i)
                graph = magicgraph.DiGraph()
                for x,y in g.edge_index.T:
                    graph[int(x)].append(int(y))
                deg = 0
                for x in graph.nodes():
                    deg += len(graph[x])
                avg_deg = deg / graph.number_of_nodes()
                graph = DoubleWeightedDiGraph(graph)
                n_nodes = graph.number_of_nodes()
                if not graph.is_connected() or n_nodes != g.x.shape[0]:
                    print('not connected')
                    not_connected += 1
                else:
                    # fc.write(str(i)+'\n')
                    deg_ls.append(avg_deg)
                    connected_indices.append(i)
                    coarsed_graphs,merged_ls = coarsening(graph, self.sfdp_path)
                    feat = g.x.numpy()
                    label_ls.append(g.y.numpy())
                    # node_nums.append(n_nodes)
                    # levels.append(len(coarsed_graphs))
                    adj_ls = [_make_adj(coarsed_graphs[0], n_nodes)]
                    for i in range(1, len(coarsed_graphs)):
                        subgraph = coarsed_graphs[i]
                        adj = _make_adj(subgraph, n_nodes)
                        adj_ls.append(adj)
                    coarsened_graph_adjs.append(np.array(adj_ls))
                    coarsened_graph_ls.append(coarsed_graphs)
                    merged_bigls.append(merged_ls)
                    feat_ls.append(feat)
            np.savetxt(self.path+'/connected.txt', connected_indices, fmt='%d')
            print('not connected:', not_connected, 'rate:', not_connected/len(graphs))
            offsets = _serialize_graph(coarsened_graph_ls,self.path+'/graphs.mtx')
            with open(self.path+'/cache','wb') as f:
                cache = {
                    'merged': merged_bigls,
                    'offsets': offsets,
                    'avg_deg': deg_ls,
                }
                pkl.dump(cache, f, protocol=4)
                print('cached')
        print('processing')
        label_ls = _enc_label(label_ls)
        degs = np.array(deg_ls)
        mask_merged,mask_node = _process_merged(merged_bigls,coarsened_graph_ls,feat_ls)
        print(len(coarsened_graph_ls), len(feat_ls), len(label_ls))
        return coarsened_graph_adjs, feat_ls, label_ls, mask_merged, mask_node, degs
