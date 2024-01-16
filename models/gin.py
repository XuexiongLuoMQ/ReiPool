import torch
import torch.nn as nn


class GINCell(nn.Module):
    def __init__(self, inp_dim, hid_dim, slope, drop, train_eps=True, eps=0., layers=2):
        super(GINCell, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(inp_dim, hid_dim),
            nn.LeakyReLU(slope),
            nn.Dropout(drop),
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(slope)
        )
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
    
    def forward(self, features, adj, mask=None):
        """
            Args:
                features: shape (N,D)
                adj: shape (N,N)
        """
        N = adj.shape[0]
        mask = torch.zeros_like(adj)
        inds = torch.nonzero(adj)
        mask[inds[:,0],inds[:,1]] = 1.0
        mask[range(N),range(N)] = 0.
        features = (1.0+self.eps) * features + torch.matmul(mask,features)
        features = self.mlp(features)
        return features

if __name__ == '__main__':
    gin = GINCell(5,128,0.2,0.5)
    print(gin.eps)
    x = torch.rand(32,5)
    adj = torch.rand(32,32)
    adj = adj * (adj > 0.5)
    output = gin(x,adj)
    print(output.shape)