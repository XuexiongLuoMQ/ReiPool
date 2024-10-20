# ReiPool

The code of this paper "ReiPool: Reinforced Pooling Graph Neural Networks for Graph-level Representation Learning", accepted by IEEE TKDE 2024.

## Research introduction
Current graph pooling methods easily destroy the global structure of the input graph during the graph pooling process and mechanically control the number of graph pooling layers for different datasets. We analyze that each graph in the graph dataset has different structures and sizes, so a specfic graph pooling strategy needs to be developed. Thus, this work aims to introduce multi-agent reinforcement learning to guide the graph pooling process to generate the most representative coarsened graph for each graph. Specifically, we design a hybrid graph coarsening module to coarsen first-order and star structures of nodes, enabling to preserve the global structure of the graph. Then, an adaptive graph pooling module with multi-agent RL is proposed to generate the most representative coarsened graph for each graph, where the feature-agent controls the fusion of coarsened node features and the depth-agent controls the number of graph coarsening layers. Finally, we design a graph-level contrast between the coarsened graph and the input graph to enhance global information preservation for graph-level representations.

### The framework of ReiPool
![](https://github.com/XuexiongLuoMQ/paper-figure/blob/master/reipool-framew.png)

## Data preparation

The datasets used in this paper are from [TUDatasets](https://chrsmrrs.github.io/datasets/docs/datasets/), which can be accessed directly through pytorch geometric.

## Requirements

    git clone https://github.com/phanein/magic-graph.git
    cd magic-graph
    python setup.py install

This code run with Python 3.
* torch == 1.10.2+cu113
* torch-geometric == 2.0.3
* torch-cluster == 1.5.9
* torch-scatter == 2.0.9
* torch-sparse == 0.6.12
* magic-graph

## Train
    python main.py --dataset PROTEINS

#### If you find that this code is useful for your research, please cite our paper:
        @ARTICLE{10689369,
        title={ReiPool: Reinforced Pooling Graph Neural Networks for Graph-Level Representation Learning},
        author={Luo, Xuexiong and Zhang, Sheng and Wu, Jia and Chen, Hongyang and Peng, Hao and Zhou, Chuan and Li, Zhao and Xue, Shan and Yang, Jian},
        journal={IEEE Transactions on Knowledge and Data Engineering},  
        pages={1-14},
        year={2024},
        }

