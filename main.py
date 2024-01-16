import torch
import argparse
import math
from gnn import gnn_env, GCN
from dqn import QAgent
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_curve,auc
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser(description='graphpooling')
parser.add_argument('--dataset',required=True)
parser.add_argument('--sfdp_path', default='bin/sfdp_linux')
parser.add_argument('--action_num', type=int, default=5)
parser.add_argument('--hid_dim', type=int, default=128)
parser.add_argument('--out_dim', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--slope', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.01) #0.01 mutag proteins DD binary,multi,collab
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--gnn_type', type=str, default='GCN')
parser.add_argument('--repeats', type=int, default=1)
parser.add_argument('--fold', type=int, default=5)
parser.add_argument('--max_timesteps', type=int, default=20)
parser.add_argument('--epoch_num', type=int, default=25)
parser.add_argument('--discount_factor', type=float, default=0.0)  #0.9 MUTAG,multi 0.8,collab 0.4
parser.add_argument('--epsilon_start', type=float, default=1.0)
parser.add_argument('--epsilon_end', type=float, default=0.05)
parser.add_argument('--epsilon_decay_steps', type=int, default=20)
parser.add_argument('--benchmark_num', type=int, default=20)
parser.add_argument('--replay_memory_size', type=int, default=10000)
parser.add_argument('--replay_memory_init_size', type=int, default=500)
parser.add_argument('--memory_batch_size', type=int, default=20)
parser.add_argument('--update_target_estimator_every', type=int, default=1)
parser.add_argument('--norm_step', type=int, default=100)
parser.add_argument('--mlp_layers', type=list, default=[128, 64, 32])
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--sample_num', type=int, default=25)
args = parser.parse_args()
args.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

def main():
    acc_records = []
    auc_records = []
    for repeat in range(args.repeats):
        print('repeat', repeat)
        env = gnn_env(dataset = args.dataset,
                     sfdp_path = args.sfdp_path,
                     max_level = args.action_num,
                     hid_dim = args.hid_dim,
                     out_dim = args.out_dim,
                     drop = args.dropout,
                     slope = args.slope,
                     lr = args.lr,
                     weight_decay = args.weight_decay,
                     gnn_type = args.gnn_type,
                     device = args.device,
                     policy = "",
                     benchmark_num = args.benchmark_num)
        
        # kf = KFold(n_splits=5, shuffle=True, random_state=1)
        kf=StratifiedKFold(n_splits=args.fold, random_state=42, shuffle=True)
        for k, (train_idx, test_idx) in enumerate(kf.split(env.net_coarsened_adj,env.net_label)):
            print(k)
            np.random.shuffle(train_idx)
            val_idx = train_idx[len(train_idx)-len(test_idx):]
            train_idx=(train_idx[:len(train_idx)-len(test_idx)])

            env.reset_train(train_idx,val_idx,test_idx)

            agent = QAgent(replay_memory_size = args.replay_memory_size,
                      replay_memory_init_size = args.replay_memory_init_size,
                      update_target_estimator_every = args.update_target_estimator_every,
                      discount_factor = args.discount_factor,
                      epsilon_start = args.epsilon_start,
                      epsilon_end = args.epsilon_end,
                      epsilon_decay_steps = args.epsilon_decay_steps,
                      lr=args.lr,
                      batch_size=args.memory_batch_size,
                      num_net = env.num_net,
                      action_num=env.action_num,
                      norm_step=args.norm_step,
                      mlp_layers=args.mlp_layers,
                      state_shape=env.state_shape,
                      device=args.device,
                      sample_num=args.sample_num)

            env.policy = agent
            agent.learn(env, args.max_timesteps)

            GNN2 = GCN(env.init_net_feat, env.net_coarsened_adj, env.mask_merged, env.mask_node,
                args.hid_dim, args.out_dim, args.dropout, args.slope, args.device, max_level=args.action_num).to(args.device)
            Optimizer = torch.optim.Adam(GNN2.parameters(), lr=0.001, weight_decay=0.0001) #0.01 mutag,0.001 proteins DD IMDB-BINARY,multi,collab

            states = []
            for i in range(len(env.init_net_feat)):
                states.append(env.get_state(i))
            states = np.stack(states)

            max_test_acc =0.

            train_num = int(np.ceil(len(train_idx)/args.batch_size))
            val_num = int(np.ceil(len(val_idx)/args.batch_size))
            test_num = int(np.ceil(len(test_idx)/args.batch_size))
            
            for epoch in range(0, args.epoch_num):
                train_loss = 0.
                val_loss = 0.
                val_pred_list = []
                val_true_list = []
                test_pred_list = []
                test_true_list = []
                print('train')
                GNN2.train()
                for i in range(0, train_num):
                    Optimizer.zero_grad()
                    train_idx1 = train_idx[args.batch_size*i:args.batch_size+args.batch_size*i]
                    train_states = states[train_idx1]
                    train_actions = np.argmax(env.best_policy.predict_nograd(train_states),axis=-1)
                    train_gnn_buffer = defaultdict(list)
                    for act, idx in zip(train_actions, train_idx1):
                        train_gnn_buffer[act].append(idx)
                    for act in range(args.action_num):
                        indexes = train_gnn_buffer[act]
                        if len(indexes) > 0:
                            preds,loss1,_ = GNN2((act,indexes))
                            labels = np.array(env.net_label[indexes])
                            labels = torch.LongTensor(labels).to(args.device)
                            loss2 = F.nll_loss(preds, labels)
                            loss=loss2+0.3*loss1 #0.7 mutag, multi 0.2, collab 0.8
                            train_loss += loss
                            loss.backward()
                            Optimizer.step()
                            print(f"trdain epoch: {epoch} step: {i}, action: {act}", "loss:", loss.item())
                GNN2.eval()
                for i in range(0, val_num):
                    val_idx1 = val_idx[args.batch_size*i:args.batch_size+args.batch_size*i]
                    val_states = states[val_idx1]
                    val_actions = np.argmax(env.best_policy.predict_nograd(val_states),axis=-1)
                    val_gnn_buffer = defaultdict(list)
                    for act, idx in zip(val_actions, val_idx1):
                        val_gnn_buffer[act].append(idx)
                    for act in range(args.action_num):
                        indexes = val_gnn_buffer[act]
                        if len(indexes) > 0:
                            preds,loss1,_= GNN2((act, indexes))
                            labels = np.array(env.net_label[indexes])
                            labels = torch.LongTensor(labels).to(args.device)
                            loss2 = F.nll_loss(preds, labels)
                            loss=loss2+0.3*loss1
                            val_loss += loss
                            preds = preds.max(1)[1]
                            val_pred_list.extend(preds.to('cpu').numpy())
                            val_true_list.extend(labels.to('cpu').numpy())
                            print(f"val epoch: {epoch} step: {i}, action: {act}", "loss:", loss.item())
                for i in range(0, test_num):
                    test_idx1 = test_idx[args.batch_size*i:args.batch_size+args.batch_size*i]
                    test_states = states[test_idx1]
                    test_actions = np.argmax(env.best_policy.predict_nograd(test_states),axis=-1)
                    test_gnn_buffer = defaultdict(list)
                    for act, idx in zip(test_actions, test_idx1):
                        test_gnn_buffer[act].append(idx)
                    for act in range(args.action_num):
                        indexes = test_gnn_buffer[act]
                        if len(indexes) > 0:
                            preds,_,_= GNN2((act, indexes))
                            labels = np.array(env.net_label[indexes])
                            labels = torch.LongTensor(labels).to(args.device)
                            preds = preds.max(1)[1]
                            test_pred_list.extend(preds.to('cpu').numpy())
                            test_true_list.extend(labels.to('cpu').numpy())
                val_pred_list = np.array(val_pred_list)
                val_true_list = np.array(val_true_list)
                val_acc = accuracy_score(val_true_list, val_pred_list)
                test_pred_list = np.array(test_pred_list)
                test_true_list = np.array(test_true_list)
                test_acc = accuracy_score(test_true_list, test_pred_list)

                print("Epoch: {}".format(epoch), " Val_Acc: {:.5f}".format(val_acc), " Test_Acc: {:.5f}".format(test_acc))
                if test_acc > max_test_acc:
                    max_test_acc = test_acc

            print("Test_Acc: {:.5f}".format(max_test_acc))
            #print("Test_AUC: {:.5f}".format(max_test_auc))
            acc_records.append(max_test_acc)
            #auc_records.append(max_test_auc)
            print('----------------------------------------------')
        mean_acc = np.mean(np.array(acc_records))
        std_acc = np.std(np.array(acc_records))
        print("Acc: {:.5f}".format(mean_acc),'± {:.5f}'.format(std_acc))
        #mean_auc = np.mean(np.array(auc_records))
        #std_auc = np.std(np.array(auc_records))
        #print("AUC: {:.5f}".format(mean_auc),'± {:.5f}'.format(std_auc))

if __name__ == '__main__':
    main()