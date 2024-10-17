from models.coarse_agent import CoarseAgent
from models.threshold_agent import ThresholdAgent
import numpy as np


class QAgent(object):
    def __init__(self,
                 replay_memory_size, replay_memory_init_size, update_target_estimator_every, discount_factor, epsilon_start, epsilon_end, epsilon_decay_steps,
                 lr, batch_size, num_net, action_num, norm_step, mlp_layers, state_shape, device, sample_num):
        self.replay_memory_size = replay_memory_size
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        self.num_net = num_net
        self.state_shape = state_shape
        self.batch_size = batch_size
        self.action_num = action_num
        self.norm_step = norm_step
        self.device = device
        self.train_t = 0
        # self.q_estimator = Estimator(action_num=action_num, lr=lr, state_shape=self.state_shape, mlp_layers=mlp_layers, device=device)
        # self.target_estimator = Estimator(action_num=action_num, lr=lr, state_shape=self.state_shape, mlp_layers=mlp_layers, device=self.device)
        # self.memory = Memory(replay_memory_size, batch_size)
        self.sample_num = sample_num
        self.coarse_agent = CoarseAgent(replay_memory_size, replay_memory_init_size, update_target_estimator_every, discount_factor, epsilon_start, epsilon_end, epsilon_decay_steps, lr, batch_size, num_net, action_num, norm_step, mlp_layers, state_shape, device, sample_num)

        self.threshold_agent = ThresholdAgent(replay_memory_size, replay_memory_init_size, update_target_estimator_every, discount_factor, epsilon_start, epsilon_end, epsilon_decay_steps, lr, batch_size, num_net, 1, norm_step, mlp_layers, state_shape, device, sample_num)


    def learn(self, env, total_timesteps):
        # env.best_policy = deepcopy(self.q_estimator)
        env.update_best_policy(self.coarse_agent, self.threshold_agent)
        
        batch_size = self.sample_num
        mem_num = 0
        last_val = 0.0
        train_idx = env.train_idx
        degrees_train = env.degrees[train_idx]
        idx_deg = list(zip(train_idx,degrees_train))
        idx_deg = sorted(idx_deg,key=lambda x:x[1],reverse=True)
        train_idx = [i[0] for i in idx_deg]
        for i in range(batch_size):
            print('batch %d' % (i+1))
            if np.random.rand() < 0.8:
                indx = train_idx[i]
            else:
                indx = np.random.choice(env.train_idx,1)[0]
            #indx = np.random.choice(env.train_idx,1)[0]
            # random.shuffle(indices)
            next_state = env.reset(indx)
            for t in range(total_timesteps):
                epsilon = self.epsilons[min(t, self.epsilon_decay_steps-1)]
                # best_action, epsilon = self.predict_batch(next_state, t)
                
                best_coarse_action = self.coarse_agent.predict_batch(next_state)[0]

                exploration_flag = np.random.choice([True, False], p=[epsilon, 1-epsilon], size=1)
                if exploration_flag:
                    best_coarse_action = np.random.choice(range(self.action_num), 1)[0]
                
                best_thresh = self.threshold_agent.predict_batch(next_state)[0]

                exploration_flag = np.random.choice([True, False], p=[epsilon, 1-epsilon], size=1)
                if exploration_flag:
                    best_thresh = np.random.rand()

                state = next_state
                
                next_state, reward, val_acc = env.step(best_coarse_action, indx, best_thresh)
                # self.memory.save(state, best_action, reward, next_state)
                self.coarse_agent.memory.save(state, best_coarse_action, reward, next_state)
                self.threshold_agent.memory.save(state, best_thresh, reward, next_state)
                mem_num += 1
                if mem_num >= self.batch_size:
                    self.train()
                if val_acc > last_val:
                    # env.best_policy = deepcopy(self.q_estimator)
                    env.update_best_policy(self.coarse_agent, self.threshold_agent)
                    last_val = val_acc
                print('learn step:', t, 'val acc:', val_acc, 'reward:', reward)
    
    def eval_step(self, states):
        coarse_actions = self.coarse_agent.eval_step(states)
        thresholds = self.threshold_agent.eval_step(states)
        return coarse_actions, thresholds
    
    # def predict_batch(self, state, t):
    #     epsilon = self.epsilons[min(t, self.epsilon_decay_steps-1)]
    #     q_values = self.q_estimator.predict_nograd(state)
    #     best_actions = np.argmax(q_values, axis=1)
    #     return best_actions, epsilon
    
    def train(self):
        self.coarse_agent.train()
        self.threshold_agent.train()
