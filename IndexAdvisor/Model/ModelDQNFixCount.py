from itertools import count
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pickle
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import Enviornment.Env3DQNFixCount as env
import Enviornment.Env3DQNFixStorage as env2
from Model import PR_Buffer as BufferX
from Model import ReplyBuffer as Buffer

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
script_name = os.path.basename(__file__)
directory = 'exp' + script_name + "mview" + '/'


class NN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(NN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            # nn.Sigmoid()
        )

    def _init_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1e-2)
                m.bias.data.uniform_(-0.1, 0.1)

    def forward(self, state):
        actions = self.layers(state)
        return actions
class DNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DNN, self).__init__()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 256)
        #self.l3 = nn.Linear(512, 256)
        self.adv1 = nn.Linear(256, 128)
        self.adv2 = nn.Linear(128, action_dim)
        self.val1 = nn.Linear(256, 128)
        self.val2 = nn.Linear(128, 1)

    def _init_weights(self):
        self.l1.weight.data.normal_(0.0, 1e-2)
        self.l1.weight.data.uniform_(-0.1, 0.1)
        self.l2.weight.data.normal_(0.0, 1e-2)
        self.l2.weight.data.uniform_(-0.1, 0.1)
        # self.l3.weight.data.normal_(0.0, 1e-2)
        # self.l3.weight.data.uniform_(-0.1, 0.1)
        self.adv1.weight.data.normal_(0.0, 1e-2)
        self.adv1.weight.data.uniform_(-0.1, 0.1)
        self.adv2.weight.data.normal_(0.0, 1e-2)
        self.adv2.weight.data.uniform_(-0.1, 0.1)
        self.val1.weight.data.normal_(0.0, 1e-2)
        self.val1.weight.data.uniform_(-0.1, 0.1)
        self.val2.weight.data.normal_(0.0, 1e-2)
        self.val2.weight.data.uniform_(-0.1, 0.1)

    def forward(self, state): # Dueling DQN
        # actions = self.layers(state)
        x = self.relu(self.l1(state))
        x = self.relu(self.l2(x))
        #x = self.relu(self.l3(x))
        adv = self.relu(self.adv1(x))
        val = self.relu(self.val1(x))
        adv = self.relu(self.adv2(adv))
        val = self.relu(self.val2(val))
        qvals = val + (adv-adv.mean())
        return qvals

class DQN:
    def __init__(self, workload, action, index_mode, conf, freq):
        self.conf = conf
        self.workload = workload
        self.action = action
        self.index_mode = index_mode

        self.state_dim = len(workload) + len(action)
        self.action_dim = len(action)

        self.actor = NN(self.state_dim, self.action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), conf['LR']) #optim.SGD(self.actor.parameters(), lr=self.conf['LR'], momentum=0.9)#

        self.replay_buffer = None
        # some monitor information
        self.num_actor_update_iteration = 0
        self.num_training = 0
        self.index_mode = index_mode
        self.actor_loss_trace = list()

        # environment
        self.envx = env.Env(self.workload, self.action, self.index_mode,  freq)


    def select_action(self, t, state):
        if not self.replay_buffer.can_update():
            action = np.random.randint(0, len(self.action))
            action = [action]
            return action
        state = torch.unsqueeze(torch.FloatTensor(state), 0) #扩展维度，按行扩展
        if np.random.randn() <= self.conf['EPISILO']:  # random policy
            self.conf['EPISILO'] -= (self.conf['EPISILO'] - 0.01) / 10000
            action = np.random.randint(0, len(self.action))
            action = [action]
            return action
        else:  # greedy policy
            self.conf['EPISILO'] -= (self.conf['EPISILO'] - 0.01) / 10000
            action_value = self.actor.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            return action

    '''def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.conf['LR'] * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr'''

    def update(self):
        for it in range(self.conf['U_ITERATION']):
            x, y, u, r, d = self.replay_buffer.sample(self.conf['BATCH_SIZE'])
            state = torch.FloatTensor(x).to(device)
            action = torch.LongTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            q_eval = self.actor(state).gather(1, action)
            q_next = self.actor(next_state).detach()
            q_target = reward + (1 - done) * self.conf['GAMMA'] * q_next.max(1)[0].view(self.conf['BATCH_SIZE'], 1)
            actor_loss = F.mse_loss(q_eval, q_target)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_loss_trace.append(actor_loss.data.item())

    def save(self):
        torch.save(self.actor.state_dict(), self.conf['NAME']+'dqn.pth')
        print('====== Model Saved ======')

    def load(self):
        print('====== Model Loaded ======')
        if os.path.exists(self.conf['NAME']+'dqn.pth'):
            self.actor.load_state_dict(torch.load(self.conf['NAME'] + 'dqn.pth', map_location='cpu'))

    def train(self, load, __x):
        #如果已存在训练好的模型，就提前导入
        if load:
            self.load()
        time_step = 0
        best_time_step = time_step
        self.envx.max_count = __x
        self.replay_buffer = Buffer.ReplayBuffer(self.conf['MEMORY_CAPACITY'], min(self.conf['LEARNING_START'],200*self.envx.max_count))
        current_best_reward = 0
        current_best_index = None
        rewards = []
        for ep in range(self.conf['EPISODES']):
            print("======"+str(ep)+"=====")
            state = self.envx.reset(__x)

            t_r = 0
            _state = []
            _next_state = []
            _action = []
            _reward = []
            _done = []
            for t in count():
                time_step += 1
                action = self.select_action(ep, state)
                # print(action)
                next_state, reward, done = self.envx.step(action)
                # print(reward)

                # if self.replay_buffer.can_update():
                #    self.update()
                t_r += reward
                self.replay_buffer.push((state, next_state, action, reward, np.float(done)))
                if t_r > current_best_reward:
                    best_time_step = time_step
                    current_best_reward = t_r
                    current_best_index = self.envx.index_trace_overall[-1]
                    current_cost_sum = self.envx.cost_trace_overall[-1]
                if done:
                        # print(current_best_index)
                    # self.replay_buffer.add(1.0, (state, next_state, action, reward, np.float(done)))
                    if self.replay_buffer.can_update() and ep % 5 == 0:
                        self.update()
                    break
                state = next_state
            rewards.append(t_r)
        self.save()
        '''plt.figure(__x)
        x = range(len(self.actor_loss_trace))
        y2 = np.array(self.actor_loss_trace)
        plt.title(self.conf['NAME'])
        plt.xlabel("Episode")
        plt.ylabel("loss")
        plt.plot(x, y2, marker='x')
        plt.savefig(self.conf['NAME'] + "loss.png", dpi=120)
        plt.clf()
        plt.close()'''
        # return self.envx.index_trace_overall[-1]
        '''with open('{}.pickles'.format(self.conf['NAME']), 'wb') as f:
            pickle.dump(self.envx.cost_trace_overall, f, protocol=0)'''
        #print("init cost_sum is:" + str(self.envx.init_cost_sum))
        self.envx.pg_client1.delete_indexes()
        indexes = []
        for _i, _idx in enumerate(current_best_index):
            if _idx == 1.0:
                indexes.append(self.envx.candidates[_i])
        for f_index in indexes:
            self.envx.pg_client1.execute_create_hypo(f_index)
        current_best_cost_sum = (
                    np.array(self.envx.pg_client1.get_queries_cost(self.envx.workload)) * self.envx.frequencies).sum()
        print("current best cost_sum is:" + str(current_best_cost_sum))
        #print("got best indexes after " + str(best_time_step) + " time step")
        #print("current best reward:" + str(current_best_reward))
        #print("performence increase: " + str((self.envx.init_cost_sum - current_best_cost_sum) / self.envx.init_cost_sum))
        #return current_best_index
        # return current_best_cost_sum
        return best_time_step


