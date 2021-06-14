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
directory = './exp' + script_name + "mview" + '/'


class DNN_Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DNN_Actor, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, action_dim)

    def _init_weights(self):
        self.l1.weight.data.normal_(0.0, 1e-2)
        self.l1.weight.data.uniform_(-0.1, 0.1)
        self.l2.weight.data.normal_(0.0, 1e-2)
        self.l2.weight.data.uniform_(-0.1, 0.1)

    def forward(self, state):
        hidden = self.relu(self.l1(state))
        softmax_input = self.relu(self.l2(hidden))
        return softmax_input


class Actor:
    def __init__(self, env, conf):
        self.time_step = 0
        self.state_dim = len(env.workload) + len(env.candidates)
        self.action_dim = len(env.candidates)
        self.cantidates = env.candidates
        self.episilo = conf['EPISILO']
        self.actor = DNN_Actor(self.state_dim, self.action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), conf['LR'])


    def select_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # 扩展维度，按行扩展
        x = self.actor.forward(state)
        prob_weights = torch.softmax(x, 1)
        arr = prob_weights.squeeze().detach().numpy()
        action = np.random.choice(range(prob_weights.shape[1]), p=arr.ravel())
        action = [action]
        return action

    def learn(self, state, action, td_error):
        action = action[0]
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        # train on episode
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # 扩展维度，按行扩展
        one_hot_action = torch.unsqueeze(torch.FloatTensor(one_hot_action), 0)  # 扩展维度，按行扩展
        softmax_input = self.actor.forward(state)
        # softmax output
        #self.actor_loss = F.cross_entropy(softmax_input, torch.argmax(one_hot_action, dim=1), reduction='none')
        self.actor_loss_ = F.cross_entropy(softmax_input, torch.argmax(one_hot_action, dim=1), reduction='none')
        td_error_ = td_error.squeeze(0)
        td_error_ = td_error_.squeeze(0)
        #self.actor_loss = -(self.actor_loss_ * td_error_).mean()
        self.actor_loss_ = -(self.actor_loss_ * td_error_.item()).mean()
        self.actor_optimizer.zero_grad()
        self.actor_loss_.backward(retain_graph=False)
        self.actor_optimizer.step()


class DNN_Critic(nn.Module):
    def __init__(self, state_dim):
        super(DNN_Critic, self).__init__()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 1)

    def _init_weights(self):
        self.l1.weight.data.normal_(0.0, 1e-2)
        self.l1.weight.data.uniform_(-0.1, 0.1)
        self.l2.weight.data.normal_(0.0, 1e-2)
        self.l2.weight.data.uniform_(-0.1, 0.1)

    def forward(self, state):
        hidden = self.relu(self.l1(state))
        q_value = self.relu(self.l2(hidden))
        return q_value

class Critic:
    def __init__(self, env, conf):
        self.time_step = 0
        self.epsilon = conf['EPISILO']
        self.state_dim = len(env.workload) + len(env.candidates)
        self.action_dim = len(env.candidates)
        self.critic = DNN_Critic(self.state_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), conf['LR'])
        self.critic_loss_trace = list()

    def train_Q_network(self, state, reward, next_state, conf):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # 扩展维度，按行扩展
        next_state = torch.unsqueeze(torch.FloatTensor(next_state), 0)  # 扩展维度，按行扩展
        q_value = self.critic.forward(state)
        next_value = self.critic.forward(next_state)
        td_error = reward + conf['GAMMA'] * next_value - q_value
        self.critic_loss = torch.square(td_error)
        self.critic_optimizer.zero_grad()
        self.critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()
        self.critic_loss_trace.append(self.critic_loss.data.item())

        return td_error

class AC:
    def __init__(self, workload, action, index_mode, conf, freq):
        self.conf = conf
        self.workload = workload
        self.action = action
        self.index_mode = index_mode

        # some monitor information
        self.num_actor_update_iteration = 0
        self.num_training = 0
        self.actor_loss_trace = list()

        # environment
        self.envx = env.Env(self.workload, self.action, self.index_mode, freq)
        # actor
        self.actor = Actor(self.envx, conf)
        # critic
        self.critic = Critic(self.envx, conf)

        # store the parameters
        self.writer = SummaryWriter(directory)

        self.learn_step_counter = 0

    def load(self):
        print('====== Model Loaded ======')
        self.actor.actor.load_state_dict(torch.load(self.conf['NAME'] + 'actor.pth', map_location='cpu'))
        self.critic.critic.load_state_dict(torch.load(self.conf['NAME'] + 'critic.pth', map_location='cpu'))

    def save(self):
        torch.save(self.actor.actor.state_dict(), self.conf['NAME'] + 'actor.pth')
        print('====== Actor Saved ======')
        torch.save(self.critic.critic.state_dict(), self.conf['NAME'] + 'critic.pth')
        print('====== Critic Saved ======')

    def train(self, load, __x):
        # 如果已存在训练好的模型，就提前导入
        if load:
            self.load()
        time_step = 0
        best_time_step = time_step
        self.envx.max_count = __x
        '''pre_create = self.envx.checkout()
        if not (pre_create is None):
            print(pre_create)
            if len(pre_create) >= __x:
                return pre_create[:__x]'''
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
                action = self.actor.select_action(state)
                # print(action)
                next_state, reward, done = self.envx.step(action)

                td_error = self.critic.train_Q_network(state, reward, next_state, self.conf)
                self.actor.learn(state, action, td_error)
                # print(reward)
                t_r += reward
                state = next_state
                if t_r > current_best_reward:
                    best_time_step = time_step
                    current_best_reward = t_r
                    current_best_index = self.envx.index_trace_overall[-1]
                if done:

                    break

            rewards.append(t_r)
        self.save()
        #Actor loss
        '''plt.figure(__x)
        x = range(len(self.critic.critic_loss_trace))
        y2 = np.array(self.critic.critic_loss_trace)
        plt.title(self.conf['NAME'])
        plt.xlabel("Episode")
        plt.ylabel("loss")
        plt.plot(x, y2, marker='x')
        plt.savefig(self.conf['NAME'] + "loss.png", dpi=120)
        plt.clf()
        plt.close()'''
        '''plt.figure(__x)
        x = range(len(self.envx.cost_trace_overall))
        y2 = [math.log(a, 10) for a in self.envx.cost_trace_overall]
        plt.plot(x, y2, marker='x')
        plt.savefig(self.conf['NAME'] + "freq.png", dpi=120)
        plt.clf()
        plt.close()
        plt.figure(__x+1)
        x = range(len(rewards))
        y2 = rewards
        plt.plot(x, y2, marker='x')
        plt.savefig(self.conf['NAME'] + "rewardfreq.png", dpi=120)
        plt.clf()
        plt.close()'''
        # return self.envx.index_trace_overall[-1]
        with open('{}.pickles'.format(self.conf['NAME']), 'wb') as f:
            pickle.dump(self.envx.cost_trace_overall, f, protocol=0)
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
        return current_best_cost_sum
