import magicgraph
from magicgraph import WeightedDiGraph, WeightedNode
import numpy as np
from collections import deque,defaultdict
import tempfile
import os
from scipy.io import mmwrite
import subprocess

class DoubleWeightedDiGraph(WeightedDiGraph):
    def __init__(self, init_graph = None):
        super(WeightedDiGraph, self).__init__(node_class=WeightedNode)
        self.weighted_nodes = magicgraph.WeightedNode()
        if init_graph is not None:
            for node, adj_list in init_graph.adjacency_iter():
                if hasattr(adj_list, 'weights'):
                    self[node].extend(adj_list, adj_list.weights)
                else:
                    self[node].extend(adj_list, [1. for adj_node in adj_list])
            if hasattr(init_graph, 'weighted_nodes'):
                self.weighted_nodes.extend(init_graph.nodes(), init_graph.weighted_nodes.weights)
            else:
                self.weighted_nodes.extend(init_graph.nodes(), [1. for node in init_graph.nodes()])
        self.visited = {node: False for node in self.nodes()}

    def is_connected(self):
        # sys.setrecursionlimit(self.number_of_nodes())
        self.visited = {node: False for node in self.nodes()}
        if self.number_of_nodes() == 0:
            return True
        self.cur_component = []
        self.bfs(list(self.nodes())[0])
        return sum(self.visited.values()) == self.number_of_nodes()

    def get_connected_components(self):
        connected_components = []
        self.visited = {node: False for node in self.nodes()}

        for node in self.nodes():
            if self.visited[node] is False:
                self.cur_component = []
                self.bfs(node)
                connected_components.append(len(self.cur_component))
        return connected_components

    # graph coarsening need to be done on each connected component
    def get_merged_connected_components(self):
        disconnected_component, connected_components, reversed_mappings = [], [], []
        self.visited = {node: False for node in self.nodes()}
        graph_size_threshold = 100

        for node in self.nodes():
            if self.visited[node] is False:
                self.cur_component = []
                self.bfs(node)
                if len(self.cur_component) >= graph_size_threshold:
                    self.cur_component = sorted(self.cur_component)
                    index_mapping = {self.cur_component[i]: i for i in range(len(self.cur_component)) }
                    connected_components.append(self.subgraph(self.cur_component, index_mapping=index_mapping))
                    reversed_mappings.append({i: self.cur_component[i] for i in range(len(self.cur_component)) })
                else:
                    disconnected_component.extend(self.cur_component)

        if len(disconnected_component) > 0:
            disconnected_component = sorted(disconnected_component)
            reversed_mappings.append({i: disconnected_component[i] for i in range(len(disconnected_component)) })
            index_mapping = {disconnected_component[i]: i for i in range(len(disconnected_component)) }
            connected_components.append(self.subgraph(disconnected_component, index_mapping=index_mapping) )
        return connected_components, reversed_mappings

    def dfs(self, cur_node):
        self.visited[cur_node] = True
        self.cur_component.append(cur_node)
        for adj_node in self[cur_node]:
            if self.visited[adj_node] is False:
                self.visited[adj_node] = True
                self.dfs(adj_node)

    def bfs(self, cur_node):
        q = deque()
        q.append(cur_node)
        self.visited[cur_node] = True

        while len(q) > 0:
            head = q.popleft()
            self.cur_component.append(head)
            for adj_node in self[head]:
                if not self.visited[adj_node]:
                    self.visited[adj_node] = True
                    q.append(adj_node)

    def subgraph(self, nodes = {}, index_mapping = None):
        nodes = set(nodes)
        if index_mapping is None:
            index_mapping = {node: node for node in nodes}
        sub = DoubleWeightedDiGraph(magicgraph.from_adjlist([ [index_mapping[node]] for node in nodes]))
        for node in nodes:
            for adj_node, weight in zip(self[node], self[node].weights):
                if adj_node in nodes:
                    sub[index_mapping[node]].append(index_mapping[adj_node], weight)
            if len(self[node]) == 0:
                if index_mapping:
                    sub[index_mapping[node]].append(index_mapping[node], 1.)
                else:
                    sub[node].append(node, 1.)

        node_weight_map = {node: weight for node, weight in zip(self.weighted_nodes, self.weighted_nodes.weights)}
        for node in nodes:
            sub.weighted_nodes.weights[index_mapping[node] ] = node_weight_map[node]
        return sub

    # get edges as pairs of integers
    def get_int_edges(self):
        edges, weights = [], []
        for node in self.nodes():
            for adj_node, weight in zip(self[node], self[node].weights):
                edges.append([node, adj_node])
                weights.append(weight)
        return edges, weights

    # get edges along with weights
    def get_edges(self):
        edges, weights = [], []
        for node in self.nodes():
            for adj_node, weight in zip(self[node], self[node].weights):
                edges.append([str(node), str(adj_node)])
                weights.append(weight)
        return edges, np.array(weights)


def load_graph(path, undirected=False):
    graph = magicgraph.load_edgelist(path, undirected=undirected)
    graph = DoubleWeightedDiGraph(graph)
    print ('Number of nodes: {}'.format(graph.number_of_nodes()))
    print ('Number of edges: {}'.format(graph.number_of_edges()))
    return graph

def coarsening(graph,sfdp_path,coarsening_scheme=2):
    # assert graph.is_connected()
    temp_dir = tempfile.mkdtemp()
    temp_fname = 'tmp.mtx'
    input_fname = os.path.join(temp_dir, temp_fname)
    print(input_fname)
    mmwrite(open(os.path.join(input_fname), 'wb'), magicgraph.to_adjacency_matrix(graph))
    sfdp_abs_path = os.path.abspath(sfdp_path)
    subprocess.call('%s -g%d -v -u -Tc %s 2>x' % (sfdp_abs_path, coarsening_scheme, input_fname), shell=True, cwd=temp_dir)
    recursive_graphs, recursive_merged_nodes = [], read_coarsening_info(temp_dir)
    cur_graph = graph
    iter_round = 1
    prev_node_count = graph.number_of_nodes()
    ec_done = False
    levels = len(recursive_merged_nodes)
    if levels == 0:
        return [graph], recursive_merged_nodes

    for level in range(levels):
        if iter_round == 1:
            print ('Original graph with %d nodes and %d edges' % \
            (cur_graph.number_of_nodes(), cur_graph.number_of_edges() ) )
            recursive_graphs.append(DoubleWeightedDiGraph(cur_graph))
        # import pdb; pdb.set_trace()
        coarsened_graph = external_collapsing(cur_graph, recursive_merged_nodes[level])
        cur_node_count = coarsened_graph.number_of_nodes()
        print ('Coarsening Round %d:' % iter_round)
        print ('Generate coarsened graph with %d nodes and %d edges' % \
        (coarsened_graph.number_of_nodes(), coarsened_graph.number_of_edges()) )

        recursive_graphs.append(coarsened_graph)
        cur_graph = coarsened_graph
        iter_round += 1
        prev_node_count = cur_node_count

    return recursive_graphs, recursive_merged_nodes

def external_collapsing(graph, merged):
    coarsened_graph = DoubleWeightedDiGraph()
    edges, weights = graph.get_int_edges()
    merged_edge_to_weight = defaultdict(float)
    node_weight = {node: weight for node, weight in zip(graph.weighted_nodes, graph.weighted_nodes.weights)}
    new_node_weights = defaultdict(float)
    # import pdb; pdb.set_trace()
    for (a, b), w in zip(edges, weights):
        if a in merged and b in merged:
            merged_a, merged_b = merged[a], merged[b]
            # if merged_a != merged_b:
            merged_edge_to_weight[(merged_a, merged_b)] += w
    for node_pair, weight in merged_edge_to_weight.items():
        coarsened_graph[node_pair[0]].append(node_pair[1], weight)
        coarsened_graph[node_pair[1]].append(node_pair[0], weight)

    for node in coarsened_graph.nodes():
        coarsened_graph.weighted_nodes.append(node, new_node_weights[node])
    return coarsened_graph.make_consistent()

def read_coarsening_info(coarsening_file_dir):
    coarsening_files = [f for dirpath, dirnames, files in os.walk(coarsening_file_dir)
        for f in files if f.startswith('prolongation')]
    levels = -1
    recursive_merged_nodes = []
    for f in coarsening_files:
        levels = max(levels, int(f[f.rfind('_') + 1:]) )
    prev_rename, rename = {}, {}
    for level in range(levels + 1):
        # different index
        merged_from = defaultdict(list)
        merged = {}
        fp = open(os.path.normpath(coarsening_file_dir) + '/' + 'prolongation_' + str(level))
        for line in fp:
            finer_node, coarser_node = map(int, line.strip().split())
            # let index starts from 0 instead
            finer_node, coarser_node = finer_node - 1, coarser_node - 1
            if finer_node in prev_rename:
                # print coarser_node, finer_node, prev_rename[finer_node]
                merged_from[coarser_node].append(prev_rename[finer_node])
            else:
                merged_from[coarser_node].append(finer_node)
        # print merged_from

        for k in merged_from.keys():
            rename[k] = merged_from[k][0]
            for node in merged_from[k]:
                merged[node] = merged_from[k][0]
        # print merged
        recursive_merged_nodes.append(merged)
        prev_rename = rename.copy()
        rename = {}
    return recursive_merged_nodes

def check_graph_pyg(dataset):
    path = 'data/'+dataset
    print('load cache')
    cache = pkl.load(open(path+'/cache','rb'))
    merged = cache['merged']
    offsets = cache['offsets']
    levels = [len(merged[i])+1 for i in range(len(merged))]
    graphs = load_dataset(path,dataset)
    connected_indices = np.loadtxt(path+'/connected.txt',dtype=int)
    node_nums = []
    for i in connected_indices:
        node_nums.append(graphs[i].x.shape[0])
    graphs = Data._deserialize_grpahs(path+'/graphs.mtx', offsets, node_nums, levels)
    print(len(graphs))
    actions = pkl.load(open(path+'/actions.pkl','rb'))
    assert len(actions) == len(graphs)
    idx_act = []
    for idx,act in actions.items():
        idx_act.append((idx,act))
    # sorted(idx_act,key=lambda x:x[0])
    acts = np.array([i[1] for i in idx_act])
    uni_acts = np.unique(acts)
    # rets = {}
    for act in uni_acts:
        if act == 0:
            continue
        # rets[act] = []
        inds = np.where(acts==act)
        ind = np.random.choice(inds,1)[0]
        ind = idx_act[ind][0]
        coars_graph = graphs[ind]
        edges_ls = []
        for graph in coars_graph:
            edges,_ = graph.get_edges()
            edges_ls.append(edges)
        draw(edges_ls,ind,merged[ind])


if __name__ == '__main__':
    import sys
    from data import Data,load_dataset
    import pickle as pkl
    from check import draw

    # graph = load_graph(sys.argv[1],True)
    # prefix = sys.argv[1].split('_')[0]
    # graphs, merged_nodes = coarsening(graph,'bin/sfdp_linux')
    # dir = f'{prefix}_coarsed_graphs'
    # if not os.path.exists(dir):
    #     os.mkdir(dir)
    # for i,graph in enumerate(graphs):
    #     with open(f'{dir}/graph{i}','w') as f:
    #         edges, weights = graph.get_edges()
    #         print(i, len(edges))
    #         for i in range(len(edges)):
    #             f.write(str(edges[i][0])+' '+str(edges[i][1])+' '+str(weights[i])+'\n') 
    check_graph_pyg(sys.argv[1])
