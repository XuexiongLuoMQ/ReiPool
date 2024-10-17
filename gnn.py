import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_coarsening import coarsening
import pandas as pd
import os
import numpy as np
import random
from data import Data
import pickle as pkl
from feats import compute_x
from collections import defaultdict
from models import gat, gin
from copy import deepcopy
# from dqn import Estimator, ThreshEstimator


class GCNCell(nn.Module):
    def __init__(self, inp_dim, hid_dim, slope, drop, layers=2,graph=False):
        super(GCNCell, self).__init__()
        self.fc1 = nn.Linear(inp_dim, hid_dim, bias=False)
        if layers >= 2:
            self.fc2 = nn.Linear(hid_dim, hid_dim, bias=False)
        else:
            self.fc2 = None
        self.leaky_relu = nn.LeakyReLU(slope)
        self.dropout = nn.Dropout(drop)
        self.graph = graph
    
    def forward(self, features, adj, mask=None):
        # features = features * mask.unsqueeze(-1)
        # print(features.shape)
        features = self.fc1(torch.matmul(adj, features))
        features = self.leaky_relu(features)
        if self.fc2:
            features = self.dropout(features)
            features = self.fc2(torch.matmul(adj, features))
            features = self.leaky_relu(features)
        # features = self.dropout(features)
        if self.graph:
            if len(features.shape) < 3:
                features = torch.mean(features,dim=0).unsqueeze(0)
            else:
                features = torch.mean(features, dim=1)
        return features

def get_gnn(gnn_type,dim_in,hid_dim,slope,drop,layers=1):
    if gnn_type == 'GCN':
        return GCNCell(dim_in,hid_dim,slope,drop,layers=layers)
    elif gnn_type == 'GAT':
        return gat.GAT(dim_in,hid_dim,slope,drop)
    elif gnn_type == 'GIN':
        return gin.GINCell(dim_in,hid_dim,slope,drop)
    else:
        raise Exception(f"Model {gnn_type} is not supported !!!")

class GCN(nn.Module):
    def __init__(self,feature_ls, adj_ls, mask_merged, mask_node, hid_dim, out_dim, drop, slope, device,sim_thres=0,drop_coars=False,gnn_type='GCN'):
        """
        Args:
            features: init graph feature, list of ndarray, shape (batch,N,D) # N is not consitent
            adjs: coarsened graph adjs, list of ndarray, shape (batch,level+1,N,N) 
            mask_merged: coarsened merged mask, list of ndarray, shape (batch,level,N,N)
            mask_node: coarsened mask, list of ndarray, shape (batch,level,N)
            sim_thres: if None, not using cosine similarity, else the threshold
            drop_coars: if True, drop the collapsed node features
        """
        super(GCN, self).__init__()
        self.sim_thres = sim_thres
        self.drop_coars = drop_coars
        # print('sim thres:',sim_thres,'drop coars:',drop_coars)
        # self.contrast = False
        # self.max_level = max_level
        # adjs = [torch.FloatTensor(adj).to(device) for adj in adj_ls]
        self.features = [torch.FloatTensor(feat).to(device) for feat in feature_ls]
        # self.masks = self.mask_process(adjs)
        adjs,masks = self.adj_process(adj_ls)
        self.adjs = [adj.to(device) for adj in adjs]
        self.masks = [mask.to(device) for mask in masks]
        self.mask_merged = [torch.FloatTensor(mask).to(device) for mask in mask_merged]
        self.mask_node = [torch.FloatTensor(mask).to(device) for mask in mask_node]
        self.dim_in = self.features[0].shape[-1]

        self.pre_n_g = nn.Sequential(nn.Linear(hid_dim, hid_dim) ,nn.ReLU(inplace=True),
                                       nn.Linear(hid_dim, hid_dim))
        self.pre_p_g = nn.Sequential(nn.Linear(hid_dim, hid_dim),nn.ReLU(inplace=True),
                                            nn.Linear(hid_dim, hid_dim))
        # self.pre_n_n = nn.Sequential(nn.Linear(hid_dim, hid_dim),nn.ReLU(inplace=True),
        #                                nn.Linear(hid_dim, hid_dim))
        # self.pre_p_n = nn.Sequential(nn.Linear(hid_dim, hid_dim),nn.ReLU(inplace=True),
        #                                     nn.Linear(hid_dim, hid_dim))
        # self.gcn = GCNCell(self.dim_in,hid_dim,slope,drop,graph=False)
        # self.gcn1 = GCNCell(hid_dim,hid_dim,slope,drop,layers=1,graph=False)
        self.gcn = get_gnn(gnn_type,self.dim_in,hid_dim,slope,drop,layers=2)
        self.gcn1 = get_gnn(gnn_type,hid_dim,hid_dim,slope,drop,layers=1)
        self.classifier = nn.Linear(hid_dim, out_dim, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
    # def mask_process(self, adj_ls):
    #     adj_num = len(adj_ls)
    #     mask_out = []
    #     for i in range(adj_num):
    #         level_num, N, _ = adj_ls[i].shape
    #         mask = torch.zeros((level_num, N)).to(adj_ls[0].device)
    #         for j in range(level_num):
    #             # adj = adj_ls[i][j]
    #             mask[j][torch.nonzero(adj_ls[i][j])[:,0].unique()] = 1.0
    #         mask_out.append(mask)
    #     return mask_out
    def adj_process(self, adjs_ls):
        adj_num = len(adjs_ls)
        adjss = []
        mask_out = []
        for k in range(adj_num):
            adjs = adjs_ls[k]
            adjs = torch.FloatTensor(adjs)
            level_num, N, _ = adjs.shape
            mask = torch.zeros((level_num,N))
            for i in range(level_num):
                mask[i][torch.nonzero(adjs[i])[:,0].unique()] = 1.0
                adjs[i] += torch.eye(N)
                adjs[i][adjs[i]>0.] = 1.
                degree_matrix = torch.sum(adjs[i], dim=-1, keepdim=False)
                degree_matrix = torch.pow(degree_matrix, -1/2)
                degree_matrix[degree_matrix == float("inf")] = 0.
                degree_matrix = torch.diag(degree_matrix)
                adjs[i] = torch.mm(degree_matrix, adjs[i])
                adjs[i] = torch.mm(adjs[i],degree_matrix)
            adjss.append(adjs)
            mask_out.append(mask)
        return adjss,mask_out
    def _aggregate(self,features,keepdim=True):
        """
        features: shape of (N,D) or (batch_size, N, D)
        """
        if len(features.shape) < 3:
            features = torch.sum(features,dim=0,keepdim=keepdim).unsqueeze(0)
        else:
            features = torch.sum(features,dim=1,keepdim=keepdim)
        return features

    def contrastive_loss_n(self,x, x_aug, temperature=0.2):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature) 
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-12)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-12)

        loss_0 = - torch.log(loss_0)
        loss_1 = - torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        return loss

    def contrastive_loss_g(self, x, x_aug, temperature=0.2):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim  / (sim_matrix.sum(dim=0) - pos_sim+ 1e-12)
        loss_1 = pos_sim  / (sim_matrix.sum(dim=1) - pos_sim+ 1e-12)

        loss_0 = - torch.log(loss_0)
        loss_1 = - torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        return loss

    def _forward(self, index, action, sim_thres):
        if isinstance(index,int) or isinstance(index, np.int64):
            #print('int index')
            init_feat = self.features[index]
            n_level = len(self.mask_merged[index]) # actual level
            level = min(action,n_level) # predicted level
            feat = self.gcn(init_feat, self.adjs[index][0], self.masks[index][0])
            feat_gl = self._aggregate(feat,keepdim=False)
            feat_ls = [self._aggregate(feat,keepdim=False)]
            for i in range(1,level+1):
                N,_ = feat.shape
                mask = self.mask_merged[index][i-1]
                mask1 = self.mask_node[index][i-1]
                coarse_feat = feat.repeat(N, 1).reshape(N,N,-1)
                feat_ = feat.repeat(1,N).reshape(N,N,-1)
                # if self.sim_thres is not None:
                sim = torch.sum(coarse_feat * feat_,dim=-1)/(torch.sqrt(torch.sum(feat_**2,dim=-1)*torch.sum(coarse_feat**2,dim=-1))+1e-6)
                sim_mask = (sim > sim_thres).float()
                mask = mask * sim_mask
                if not self.drop_coars:
                    coarse_feat = torch.sum(coarse_feat * mask.unsqueeze(-1),dim=1)
                    coarse_feat = coarse_feat + feat * (self.masks[index][i]-mask1).unsqueeze(-1)
                else:
                    coarse_feat = feat * self.masks[index][i].unsqueeze(-1)
                feat_ls.append(self._aggregate(coarse_feat,keepdim=False))
                feat = self.gcn1(coarse_feat,self.adjs[index][i])
                # feat_ls.append(self._aggregate(feat,keepdim=False))
            # TODO
            graph_feat=self._aggregate(feat,keepdim=False)
            return feat_gl,graph_feat,torch.cat(feat_ls,dim=0) # (level,D)
        else:
            graph_feats = []
            feat_g=[]
            loss=0
            feat_coarss = []
            for i in range(len(index)):
                feat_gl,graph_feat,feat_coars=self._forward(index[i],action,sim_thres[i])
                graph_feats.append(graph_feat)
                feat_g.append(feat_gl)
                feat_coarss.append(feat_coars)
            return torch.cat(feat_g,dim=0), torch.cat(graph_feats,dim=0), feat_coarss  # (batch, D)
            
    def forward(self, input):
        action, index, sim_thres = input
        feat_gg,feats,feats_coarss = self._forward(index,action, sim_thres)
        feat_gg=self.pre_n_g(feat_gg)
        feat=self.pre_p_g(feats)
        loss2=self.contrastive_loss_g(feat,feat_gg)
        loss2=loss2.mean()
        loss=loss2
        predict = self.classifier(feat)
        predict = F.log_softmax(predict, dim=1)
        return predict,loss,feats_coarss

class gnn_env(object):
    def __init__(self, dataset, sfdp_path, max_level, hid_dim, out_dim, drop, slope, lr, weight_decay, gnn_type, device, policy, benchmark_num):
       self.path = 'data/'+dataset
       self.dataset = dataset
       self.sfdp_path = sfdp_path
       self.action_num = max_level
       self.device = device
       self.policy = policy
       self.benchmark_num = benchmark_num
       # model parameters
       self.gnn_type = gnn_type
       self.hid_dim = hid_dim
    #    self.out_dim = out_dim
       self.drop = drop
       self.slope = slope
       # optimizer
       self.lr = lr
       self.weight_decay = weight_decay
        # load data
       self.data = Data(self.path,self.dataset,self.sfdp_path)
       self.load_dataset()

    def load_dataset(self):
        #ratio = [0.8,0.1,0.1]
        self.net_coarsened_adj, self.init_net_feat, \
            self.net_label,self.mask_merged,self.mask_node,self.degrees,n_classes = self.data.load()  # graph feat and labels of shape (batch_size,N,D), (batch_size,)
        self.num_net = len(self.net_label)
        self.out_dim = n_classes
        print('num classes', n_classes)

    def padding_state(self, state):
        """
        state: shape of (N,)
        """
        assert state.shape[-1] <= self.hid_dim
        if state.shape[-1] == self.hid_dim:
            return state
        padding = np.zeros((*state.shape[:-1],self.hid_dim-state.shape[-1]))
        padding_state = np.concatenate([state,padding],axis=-1)
        return padding_state

    def get_state(self,indx):
        state = self.padding_state(np.mean(self.init_net_feat[indx],axis=0))
        return state

    def reset(self,indx):
        state = self.get_state(indx)
        self.optimizer.zero_grad()
        return state

    def reset_train(self, train_idx, val_idx, test_idx,sim_thres=0,drop_coars=False):
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.num_train = len(train_idx)
        self.num_val = len(val_idx)
        self.num_test = len(test_idx)
        # if self.gnn_type == 'GCN':
        #    self.model = GCN(self.init_net_feat, self.net_coarsened_adj, self.mask_merged, self.mask_node,
        #     self.hid_dim, self.out_dim, self.drop, self.slope, self.device, max_level=self.action_num).to(self.device)
        # else:
        #     raise Exception(f"Model {self.gnn_type} is not supported !!!")
        # self.init_gcn = get_gnn(self.gnn_type,self.init_net_feat[0].shape[-1],self.hid_dim,self.slope,self.drop).to(self.device)
        self.model = GCN(self.init_net_feat, self.net_coarsened_adj, self.mask_merged, self.mask_node,
            self.hid_dim, self.out_dim, self.drop, self.slope, self.device, sim_thres, drop_coars, self.gnn_type).to(self.device)
        print(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.batch_size_qdn = self.num_train
        self.state_shape = self.get_state(0).shape
        # self.gnn_buffers = defaultdict(list)
        self.past_performance = [0.]

    def transition(self, feat_coars,index):
        # next_index = np.random.choice(self.train_idx,1)[0]
        # next_state = np.mean(self.init_net_feat[next_index],axis=0)
        # level = min(action, len(feat_coars)-1)
        next_state = feat_coars[-1]
        return next_state
    
    def step(self, action, index, thres):
        feat_coars = self.train(action, index, thres)
        next_state = self.transition(feat_coars, index)
        val_acc = self.eval()
        benchmark = np.mean(np.array(self.past_performance[-self.benchmark_num:]))
        self.past_performance.append(val_acc)
        reward = val_acc - benchmark
        return next_state.data.cpu().numpy(), reward,val_acc
    
    def train(self, act, index, thres):
        self.model.train()
        self.optimizer.zero_grad()
        pred,loss0,feat_coars = self.model((act, index, thres))
        label = np.array([self.net_label[index]])
        label = torch.LongTensor(label).to(self.device)
        loss=F.nll_loss(pred, label)#+loss0
        loss.backward()
        self.optimizer.step()
        return feat_coars
    
    def eval(self):
        self.model.eval()
        batch_dict = {}
        val_indexes = self.val_idx
        val_states = []
        for i in val_indexes:
            val_states.append(self.get_state(i))
        val_states = np.stack(val_states)
        coarse_actions, thresholds = self.policy.eval_step(val_states)
        for act, idx, thres in zip(coarse_actions, val_indexes, thresholds):
            if act not in batch_dict.keys():
                batch_dict[act] = []
            batch_dict[act].append((idx,thres))
        val_acc = 0.
        for act in batch_dict.keys():
            indexes,threshs = map(list, zip(*batch_dict[act]))
            # import pdb; pdb.set_trace()
            if len(indexes) > 0:
                preds,_ ,_= self.model((act, indexes, threshs))
                preds = preds.max(1)[1]
                labels = torch.LongTensor(self.net_label[indexes]).to(self.device)
                val_acc += preds.eq(labels).sum().item()
        return val_acc/len(val_indexes)

    def update_best_policy(self, coarse_agent, thresh_agent):
        self.coarse_policy = deepcopy(coarse_agent.q_estimator)
        self.thresh_policy = deepcopy(thresh_agent.q_estimator)

    def best_policy_predict(self, states):
        best_coarse_actions = self.coarse_policy.predict_nograd(states)
        best_thresholds = self.thresh_policy.predict_nograd(states)
        return best_coarse_actions, best_thresholds
    

    def save_best_policy(self, output_dir,fold=0):
        torch.save(self.coarse_policy, os.path.join(output_dir, f'policy_coarse_{fold}.pt'))
        torch.save(self.thresh_policy, os.path.join(output_dir, f'policy_thresh_{fold}.pt'))

    
    def load_best_policy(self, checkpoint_dir,fold=0):
        self.coarse_policy = torch.load(os.path.join(checkpoint_dir, f"policy_coarse_{fold}.pt"))
        self.thresh_policy = torch.load(os.path.join(checkpoint_dir, f"policy_thresh_{fold}.pt"))
    
        
