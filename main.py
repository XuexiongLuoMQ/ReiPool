import torch
import argparse
import math
from gnn import gnn_env, GCN
from agent_chain import QAgent
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import pickle as pkl
import os
from copy import deepcopy


parser = argparse.ArgumentParser(description="graphpooling")
parser.add_argument("--dataset", required=True)
parser.add_argument("--sfdp_path", default="bin/sfdp_linux")
parser.add_argument("--action_num", type=int, default=5)
parser.add_argument("--hid_dim", type=int, default=128)
parser.add_argument("--out_dim", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--slope", type=float, default=0.2)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--weight_decay", type=float, default=0.0001)
parser.add_argument("--gnn_type", type=str, default="GAT")
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--fold", type=int, default=5)
parser.add_argument("--max_timesteps", type=int, default=15)#%%%
parser.add_argument("--epoch_num", type=int, default=50)
parser.add_argument("--discount_factor", type=float, default=0.1)
parser.add_argument("--epsilon_start", type=float, default=1.0)
parser.add_argument("--epsilon_end", type=float, default=0.05)
parser.add_argument("--epsilon_decay_steps", type=int, default=20)
parser.add_argument("--benchmark_num", type=int, default=25)
parser.add_argument("--replay_memory_size", type=int, default=10000)
parser.add_argument("--replay_memory_init_size", type=int, default=500)
parser.add_argument("--memory_batch_size", type=int, default=20)
parser.add_argument("--update_target_estimator_every", type=int, default=1)
parser.add_argument("--norm_step", type=int, default=100)
parser.add_argument("--mlp_layers", type=list, default=[128, 64, 32])
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--sample_num", type=int, default=30)
parser.add_argument("--beta", type=float, default=0.6)
parser.add_argument(
    "--sim_thres", type=float, default=0, help="cosine similarity threshold"
)
parser.add_argument(
    "--drop_coars",
    action="store_true",
    help="whether to drop the collapse node feature",
)
parser.add_argument("--n_fold", type=int, default=5)
parser.add_argument("--test", action="store_true")

args = parser.parse_args()
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test():
    checkpoint_root = f"checkpoint/{args.dataset}"
    with open(f"{checkpoint_root}/args.pkl", "rb") as f:
        args1 = pkl.load(f)
    print(args1)
    test_accs = []
    env = gnn_env(
        dataset=args1.dataset,
        sfdp_path=args1.sfdp_path,
        max_level=args1.action_num,
        hid_dim=args1.hid_dim,
        out_dim=args1.out_dim,
        drop=args1.dropout,
        slope=args1.slope,
        lr=args1.lr,
        weight_decay=args1.weight_decay,
        gnn_type=args1.gnn_type,
        device=args1.device,
        policy="",
        benchmark_num=args1.benchmark_num,
    )
    device = args1.device
    # estimator = Estimator(args1.action_num,args1.lr,(args1.hid_dim,),args1.mlp_layers,device)
    GNN = GCN(
        env.init_net_feat,
        env.net_coarsened_adj,
        env.mask_merged,
        env.mask_node,
        args1.hid_dim,
        env.out_dim,
        args1.dropout,
        args1.slope,
        args1.device,
        args1.sim_thres,
        args1.drop_coars,
        args1.gnn_type,
    ).to(device)

    best_acc = 0
    coars_times = {}
    all_states = []
    for i in range(len(env.net_label)):
        all_states.append(env.get_state(i))

    for k in range(args1.n_fold):
        test_idx = np.loadtxt(
            f"{checkpoint_root}/{args1.n_fold}fold_idx/test_idx-{k+1}.txt", dtype=int
        )
        ckpt = torch.load(f"{checkpoint_root}/models_{k}.pt")
        # estimator.qnet.load_state_dict(ckpt['qnet'])
        env.load_best_policy(checkpoint_root, k)
        GNN.load_state_dict(ckpt["gnn"])
        states = []
        for i in test_idx:
            states.append(all_states[i])
        states = np.stack(states)
        test_num = int(np.ceil(len(test_idx) / args1.batch_size))
        test_pred_list = []
        test_true_list = []
        GNN.eval()
        for i in range(0, test_num):
            test_idx1 = test_idx[args1.batch_size * i : (i + 1) * args1.batch_size]
            test_states = states[args1.batch_size * i : (i + 1) * args1.batch_size]
            # test_actions = np.argmax(estimator.predict_nograd(test_states),axis=-1)
            coarse_actions, thresholds = env.best_policy_predict(test_states)
            coarse_actions = np.argmax(coarse_actions, axis=-1)
            thresholds = thresholds.flatten()
            test_gnn_buffer = defaultdict(list)
            for act, idx, thres in zip(coarse_actions, test_idx1, thresholds):
                test_gnn_buffer[act].append((idx, thres))
            for act in range(args1.action_num):
                indexes = test_gnn_buffer[act]
                if len(indexes) > 0:
                    indexes, threshs = map(list, zip(*indexes))
                    with torch.no_grad():
                        preds, _, _ = GNN((act, indexes, threshs))
                    labels_ = np.array(env.net_label[indexes])
                    labels_ = torch.LongTensor(labels_).to(device)
                    preds = preds.max(1)[1]
                    test_pred_list.extend(preds.cpu().numpy())
                    test_true_list.extend(labels_.cpu().numpy())
        test_pred_list = np.array(test_pred_list)
        test_true_list = np.array(test_true_list)
        test_acc = accuracy_score(test_true_list, test_pred_list)
        test_accs.append(test_acc)
        print(k, test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            batch_size = 64
            for i in range(0, len(all_states), batch_size):
                # actions = np.argmax(estimator.predict_nograd(np.stack(all_states[i:i+batch_size])),axis=-1)
                coarse_actions, thresholds = env.best_policy_predict(
                    np.stack(all_states[i : i + batch_size])
                )
                actions = np.argmax(coarse_actions, axis=-1)
                thresholds = thresholds.flatten()
                for idx, (act, thres) in enumerate(zip(actions, thresholds)):
                    coars_times[i + idx] = (act, thres)
    mean_acc = np.mean(np.array(test_accs))
    std_acc = np.std(np.array(test_accs))
    print("Acc: {:.5f} ± {:.5f}".format(mean_acc, std_acc))
    with open(f"{checkpoint_root}/actions.pkl", "wb") as f:
        pkl.dump(coars_times, f)


def main():
    logf = open("%s_results.txt" % args.dataset, "a")
    checkpoint_root = f"checkpoint/{args.dataset}"
    if not os.path.exists(checkpoint_root):
        os.makedirs(checkpoint_root)
    with open(f"{checkpoint_root}/args.pkl", "wb") as f:
        pkl.dump(args, f)
    acc_records = []
    auc_records = []
    for repeat in range(args.repeats):
        print("repeat", repeat)
        logf.write("repeat" + str(repeat) + "\n")
        env = gnn_env(
            dataset=args.dataset,
            sfdp_path=args.sfdp_path,
            max_level=args.action_num,
            hid_dim=args.hid_dim,
            out_dim=args.out_dim,
            drop=args.dropout,
            slope=args.slope,
            lr=args.lr,
            weight_decay=args.weight_decay,
            gnn_type=args.gnn_type,
            device=args.device,
            policy="",
            benchmark_num=args.benchmark_num,
        )

        # kf = KFold(n_splits=5, shuffle=True, random_state=1)
        n_fold = args.n_fold
        if not os.path.exists(f"{checkpoint_root}/{n_fold}fold_idx"):
            os.mkdir(f"{checkpoint_root}/{n_fold}fold_idx")
        kf = StratifiedKFold(n_splits=n_fold, random_state=42, shuffle=True)
        for k, (train_idx, test_idx) in enumerate(
            kf.split(env.net_coarsened_adj, env.net_label)
        ):
            print(k)
            np.random.shuffle(train_idx)
            val_idx = train_idx[len(train_idx) - len(test_idx) :]
            train_idx = train_idx[: len(train_idx) - len(test_idx)]
            np.savetxt(
                f"{checkpoint_root}/{n_fold}fold_idx/train_idx-{k+1}.txt",
                train_idx,
                fmt="%d",
            )
            np.savetxt(
                f"{checkpoint_root}/{n_fold}fold_idx/val_idx-{k+1}.txt",
                val_idx,
                fmt="%d",
            )
            np.savetxt(
                f"{checkpoint_root}/{n_fold}fold_idx/test_idx-{k+1}.txt",
                test_idx,
                fmt="%d",
            )

            env.reset_train(
                train_idx, val_idx, test_idx, args.sim_thres, args.drop_coars
            )

            agent = QAgent(
                replay_memory_size=args.replay_memory_size,
                replay_memory_init_size=args.replay_memory_init_size,
                update_target_estimator_every=args.update_target_estimator_every,
                discount_factor=args.discount_factor,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                epsilon_decay_steps=args.epsilon_decay_steps,
                lr=args.lr,
                batch_size=args.memory_batch_size,
                num_net=env.num_net,
                action_num=env.action_num,
                norm_step=args.norm_step,
                mlp_layers=args.mlp_layers,
                state_shape=env.state_shape,
                device=args.device,
                sample_num=args.sample_num,
            )

            env.policy = agent
            agent.learn(env, args.max_timesteps)

            GNN2 = GCN(
                env.init_net_feat,
                env.net_coarsened_adj,
                env.mask_merged,
                env.mask_node,
                args.hid_dim,
                env.out_dim,
                args.dropout,
                args.slope,
                args.device,
                args.sim_thres,
                args.drop_coars,
                args.gnn_type,
            ).to(args.device)
            Optimizer = torch.optim.Adam(
                GNN2.parameters(), lr=0.001, weight_decay=0.0001
            )

            states = []
            for i in range(len(env.init_net_feat)):
                states.append(env.get_state(i))
            states = np.stack(states)

            max_test_acc = 0.0

            train_num = int(np.ceil(len(train_idx) / args.batch_size))
            val_num = int(np.ceil(len(val_idx) / args.batch_size))
            test_num = int(np.ceil(len(test_idx) / args.batch_size))

            for epoch in range(0, args.epoch_num):
                train_loss = 0.0
                val_loss = 0.0
                val_pred_list = []
                val_true_list = []
                test_pred_list = []
                test_true_list = []
                print("train")
                GNN2.train()
                for i in range(0, train_num):
                    Optimizer.zero_grad()
                    train_idx1 = train_idx[
                        args.batch_size * i : args.batch_size + args.batch_size * i
                    ]
                    train_states = states[train_idx1]

                    # train_actions = np.argmax(env.best_policy.predict_nograd(train_states),axis=-1)
                    coarse_actions, thresholds = env.best_policy_predict(train_states)
                    coarse_actions = np.argmax(coarse_actions, axis=-1)
                    thresholds = thresholds.flatten()
                    train_gnn_buffer = defaultdict(list)
                    for act, thres, idx in zip(coarse_actions, thresholds, train_idx1):
                        train_gnn_buffer[act].append((idx, thres))
                    for act in range(args.action_num):
                        indexes = train_gnn_buffer[act]
                        if len(indexes) > 0:
                            indexes, threshs = map(list, zip(*indexes))
                            preds, loss1, _ = GNN2((act, indexes, threshs))
                            labels = np.array(env.net_label[indexes])
                            labels = torch.LongTensor(labels).to(args.device)
                            loss2 = F.nll_loss(preds, labels)
                            loss = loss2 + args.beta*loss1
                            train_loss += loss
                            loss.backward()
                            Optimizer.step()
                            print(
                                f"train epoch: {epoch} step: {i}, action: {act}",
                                "loss:",
                                loss.item(),
                            )
                GNN2.eval()
                for i in range(0, val_num):
                    val_idx1 = val_idx[
                        args.batch_size * i : args.batch_size + args.batch_size * i
                    ]
                    val_states = states[val_idx1]
                    coarse_actions, thresholds = env.best_policy_predict(val_states)
                    coarse_actions = np.argmax(coarse_actions, axis=-1)
                    thresholds = thresholds.flatten()
                    # val_actions = np.argmax(env.best_policy.predict_nograd(val_states),axis=-1)
                    val_gnn_buffer = defaultdict(list)
                    for act, idx, thres in zip(coarse_actions, val_idx1, thresholds):
                        val_gnn_buffer[act].append((idx, thres))
                    for act in range(args.action_num):
                        indexes = val_gnn_buffer[act]
                        if len(indexes) > 0:
                            indexes, threshs = map(list, zip(*indexes))
                            preds, loss1, _ = GNN2((act, indexes, threshs))
                            labels = np.array(env.net_label[indexes])
                            labels = torch.LongTensor(labels).to(args.device)
                            loss2 = F.nll_loss(preds, labels)
                            loss = loss2 + args.beta*loss1
                            val_loss += loss
                            preds = preds.max(1)[1]
                            val_pred_list.extend(preds.to("cpu").numpy())
                            val_true_list.extend(labels.to("cpu").numpy())
                            print(
                                f"val epoch: {epoch} step: {i}, action: {act}",
                                "loss:",
                                loss.item(),
                            )
                for i in range(0, test_num):
                    test_idx1 = test_idx[
                        args.batch_size * i : args.batch_size + args.batch_size * i
                    ]
                    test_states = states[test_idx1]
                    coarse_actions, thresholds = env.best_policy_predict(test_states)
                    coarse_actions = np.argmax(coarse_actions, axis=-1)
                    thresholds = thresholds.flatten()
                    # test_actions = np.argmax(env.best_policy.predict_nograd(test_states),axis=-1)
                    test_gnn_buffer = defaultdict(list)
                    for act, idx, thres in zip(coarse_actions, test_idx1, thresholds):
                        test_gnn_buffer[act].append((idx, thres))
                    for act in range(args.action_num):
                        indexes = test_gnn_buffer[act]
                        if len(indexes) > 0:
                            indexes, threshs = map(list, zip(*indexes))
                            preds, _, _ = GNN2((act, indexes, threshs))
                            labels = np.array(env.net_label[indexes])
                            labels = torch.LongTensor(labels).to(args.device)
                            preds = preds.max(1)[1]
                            test_pred_list.extend(preds.to("cpu").numpy())
                            test_true_list.extend(labels.to("cpu").numpy())
                val_pred_list = np.array(val_pred_list)
                val_true_list = np.array(val_true_list)
                val_acc = accuracy_score(val_true_list, val_pred_list)
                test_pred_list = np.array(test_pred_list)
                test_true_list = np.array(test_true_list)
                test_acc = accuracy_score(test_true_list, test_pred_list)

                print(
                    "Epoch: {}".format(epoch),
                    " Val_Acc: {:.5f}".format(val_acc),
                    " Test_Acc: {:.5f}".format(test_acc),
                )
                if test_acc > max_test_acc:
                    max_test_acc = test_acc
                    best_gnn = deepcopy(GNN2.state_dict())
            torch.save({"gnn": best_gnn}, f"{checkpoint_root}/models_{k}.pt")
            env.save_best_policy(checkpoint_root, k)
            print("Test_Acc: {:.5f}".format(max_test_acc))
            logf.write("Fold {}, Test_Acc: {:.5f}\n".format(k, max_test_acc))
            # print("Test_AUC: {:.5f}".format(max_test_auc))
            acc_records.append(max_test_acc)
            # auc_records.append(max_test_auc)
            print("----------------------------------------------")
        mean_acc = np.mean(np.array(acc_records))
        std_acc = np.std(np.array(acc_records))
        print("Acc: {:.5f}".format(mean_acc), "± {:.5f}".format(std_acc))
        logf.write("----------------------------------------------\n")
        logf.write("Acc: {:.5f} ± {:.5f}\n".format(mean_acc, std_acc))
        # mean_auc = np.mean(np.array(auc_records))
        # std_auc = np.std(np.array(auc_records))
        # print("AUC: {:.5f}".format(mean_auc),'± {:.5f}'.format(std_auc))
    logf.close()


if __name__ == "__main__":
    if args.test:
        test()
    else:
        main()
