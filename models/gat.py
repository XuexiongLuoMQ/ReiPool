import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout       = dropout        # drop prob = 0.6
        self.in_features   = in_features    # 
        self.out_features  = out_features   # 
        self.alpha         = alpha          # LeakyReLU with negative input slope, alpha = 0.2
        self.concat        = concat         # conacat = True for all layers except the output layer.

        
        # Xavier Initialization of Weights
        # Alternatively use weights_init to apply weights of choice 
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        """
        Args:
            input: tensor of shape (N,D)
            adj: tensor of shape (N, N)
        """
        # Linear Transformation
        h = torch.mm(input, self.W) # matrix multiplication
        N = h.size()[0]
        # print(N)

        # Attention Mechanism
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # Masked Attention
        zero_vec  = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime   = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid, alpha=0.2, dropout=0.5, nheads=1):
#         """Dense version of GAT.
#         Args:    
#             nfeat: input feat size
#             nhid: hidden size
#             dropout: dropout rate
#             alpha: alpha of leakyReLU
#             nheads: number of heads
#         return:
#             concatenated multi-head features
#         """
#         super(GAT, self).__init__()
#         self.dropout = dropout

#         self.attentions = [GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)] 
#          #输入到隐藏层
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)

#         # self.out_att = GATLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
#         #multi-head 隐藏层到输出

#     def forward(self, x, adj, mask=None):
#         # x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         # x = F.dropout(x, self.dropout, training=self.training)
#         # x = F.elu(self.out_att(x, adj))
#         return x

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, alpha=0.2, dropout=0.5, nheads=1):
        """Dense version of GAT.
        Args:    
            nfeat: input feat size
            nhid: hidden size
            dropout: dropout rate
            alpha: alpha of leakyReLU
            nheads: number of heads
        return:
            concatenated multi-head features
        """
        super(GAT, self).__init__()
        self.gat = GATConv(nfeat,nhid,nheads,concat=True,negative_slope=alpha,dropout=dropout)
    
    def forward(self, x, adj, mask=None):
        inds = torch.nonzero(adj)
        new_inds = torch.unique(inds[:,0])
        feat = x[new_inds]
        edge_index = []
        new_inds = new_inds.cpu().numpy()
        inds = inds.cpu().numpy()
        inds2new = {k:i for i,k in enumerate(new_inds)}
        for i in range(len(inds)):
            edge_index.append([inds2new[inds[i,0]],inds2new[inds[i,1]]])
        edge_index = torch.tensor(edge_index,dtype=torch.long).t().contiguous().to(x.device)
        output = self.gat(feat,edge_index)
        output = F.elu(output)

        return output