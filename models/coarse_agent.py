import numpy as np
from dqn import Memory, Estimator
from copy import deepcopy


class CoarseAgent(object):
    def __init__(self,
                 replay_memory_size, replay_memory_init_size, update_target_estimator_every, discount_factor, epsilon_start, epsilon_end, epsilon_decay_steps,
                 lr, batch_size, num_net, action_num, norm_step, mlp_layers, state_shape, device, sample_num):
        self.replay_memory_size = replay_memory_size
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        # self.epsilon_decay_steps = epsilon_decay_steps
        # self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        # self.num_net = num_net
        self.state_shape = state_shape
        self.batch_size = batch_size
        self.action_num = action_num
        self.norm_step = norm_step
        self.device = device
        self.train_t = 0
        self.q_estimator = Estimator(action_num=action_num, lr=lr, state_shape=self.state_shape, mlp_layers=mlp_layers, device=device)
        self.target_estimator = Estimator(action_num=action_num, lr=lr, state_shape=self.state_shape, mlp_layers=mlp_layers, device=self.device)
        self.memory = Memory(replay_memory_size, batch_size)
        self.sample_num = sample_num
    
    # def learn(self, env, total_timesteps):
    #     env.best_policy = deepcopy(self.q_estimator)
    #     batch_size = self.sample_num
    #     mem_num = 0
    #     last_val = 0.0
    #     train_idx = env.train_idx
    #     degrees_train = env.degrees[train_idx]
    #     idx_deg = list(zip(train_idx,degrees_train))
    #     idx_deg = sorted(idx_deg,key=lambda x:x[1],reverse=True)
    #     train_idx = [i[0] for i in idx_deg]
    #     for i in range(batch_size):
    #         print('batch %d' % (i+1))
    #         if np.random.rand() < 0.8:
    #             indx = train_idx[i]
    #         else:
    #             indx = np.random.choice(env.train_idx,1)[0]
    #         #indx = np.random.choice(env.train_idx,1)[0]
    #         # random.shuffle(indices)
    #         next_state = env.reset(indx)
    #         for t in range(total_timesteps):
    #             best_action, epsilon = self.predict_batch(next_state, t)
    #             best_action = best_action[0]
    #             exploration_flag = np.random.choice([True, False], p=[epsilon, 1-epsilon], size=1)
    #             if exploration_flag:
    #                 best_action = np.random.choice(range(self.action_num), 1)[0]
    #             state = next_state
    #             next_state, reward, val_acc = env.step(best_action, indx)
    #             self.memory.save(state, best_action, reward, next_state)
    #             mem_num += 1
    #             if mem_num >= self.batch_size:
    #                 self.train()
    #             if val_acc > last_val:
    #                 env.best_policy = deepcopy(self.q_estimator)
    #                 last_val = val_acc
    #             print('learn step:', t, 'val acc:', val_acc, 'reward:', reward)
    
    def eval_step(self, states):
        q_values = self.q_estimator.predict_nograd(states)
        best_actions = np.argmax(q_values, axis=-1)
        return best_actions
    
    def predict_batch(self, state):
        q_values = self.q_estimator.predict_nograd(state)
        best_actions = np.argmax(q_values, axis=1)
        
        return best_actions
    
    def train(self):
        state_batch, action_batch, reward_batch, next_state_batch = self.memory.sample()
        q_values_next_target = self.target_estimator.predict_nograd(next_state_batch)
        best_actions_next = np.argmax(q_values_next_target, axis=1)
        target_batch = reward_batch + self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions_next]
        self.q_estimator.update(state_batch, action_batch, target_batch)
        self.target_estimator = deepcopy(self.q_estimator)