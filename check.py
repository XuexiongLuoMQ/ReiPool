import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

def make_undirected(edges):
    es = set()
    res = []
    for i in range(len(edges)):
        x,y = int(edges[i][0]),int(edges[i][1])
        if (y,x) not in es:
            res.append((x,y))
            es.add((x,y))
    return res

def draw(edges_ls, index, merged=None):
    plt.figure(figsize=(8,12))
    for i in range(len(edges_ls)):
        edges = edges_ls[i]
        edges = make_undirected(edges)
        print(f'graph{index}-{i}, {len(edges)} edges')
        G = nx.Graph()
        G.add_edges_from(edges)
        # pos = nx.circular_layout(G)
        plt.subplot(2,3,i+1)
        nx.draw(G, with_labels=True)
        plt.title(f'graph{index}-{i}')
        if i>0:
            print('merged:', merged[i-1])
    plt.savefig(f'graph{index}-{i}.png',dpi=300,format='png')
    plt.show()


if __name__ == '__main__':
    import sys
    edges = pd.read_csv(sys.argv[1],sep=' ',header=None).values
    merged = open(sys.argv[2]).readlines() if len(sys.argv)>2 else None