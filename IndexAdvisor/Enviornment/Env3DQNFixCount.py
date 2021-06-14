import numpy as np
from Utility import PostgreSQL as pg
import math
from typing import List
import sys


class Env:
    def __init__(self, workload, candidates, mode,  freq=None, query2cands=None, dicts=None):
        self.workload = workload
        self.candidates = candidates
        # create real/hypothetical index
        self.mode = mode
        self.pg_client1 = pg.PGHypo()
        self.pg_client2 = pg.PGHypo()
        '''self._frequencies = freq
        self.frequencies = np.array(self._frequencies) / np.array(self._frequencies).sum()'''
        self.frequencies = freq  # 最后一组索引更新测试直接用这个
        # 用一个字典保存每个索引的频率因子<index,freq>
        # 注意：有一些组合索引有可能不存在于query中，这些索引的频率用平均值填入
        self.index2freq_dict = {}
        if query2cands != None:
            for i in range(len(self.workload)):
                cands = query2cands[i]  # 取出当前查询的候选索引
                for index in cands:
                    if (index in self.index2freq_dict.keys()):
                        freq = self.index2freq_dict[index] * 0.1
                        freq += self.frequencies[i]
                        self.index2freq_dict[index] = freq
                    else:
                        self.index2freq_dict[index] = self.frequencies[i]
        self.dicts = dicts
        # state info
        self.init_cost = np.array(self.pg_client1.get_queries_cost(workload))*self.frequencies
        self.init_cost_sum = self.init_cost.sum() #未建任何索引
        # self.init_state = np.append(self.init_cost, np.zeros((len(candidates),), dtype=np.float))
        self.init_state = np.append(self.frequencies, np.zeros((len(candidates),), dtype=np.float))
        self.last_state = self.init_state
        self.last_cost = self.init_cost
        self.last_cost_sum = self.init_cost_sum

        # utility info
        self.index_oids = np.zeros((len(candidates),), dtype=np.int)
        self.performance_gain = np.zeros((len(candidates),), dtype=np.float)
        self.current_index_count = 0
        self.currenct_index = np.zeros((len(candidates),), dtype=np.float)
        self.current_index_storage = np.zeros((len(candidates),), dtype=np.float)

        # monitor info
        self.cost_trace_overall = list()
        self.index_trace_overall = list()
        self.min_cost_overall = list()
        self.min_indexes_overall = list()
        self.current_min_cost = (np.array(self.pg_client1.get_queries_cost(workload))*0.1*self.frequencies).sum()
        self.current_min_index = np.zeros((len(candidates),), dtype=np.float)

        self.current_storage_sum = 0
        self.max_count = 0

        self.pre_create = []
        self.imp_count = 0

    def checkout(self):
        pre_is = []
        while True:
            current_max = 0
            current_index = None
            current_index_len = 0
            # 每次start_sum会基于当前的索引配置更新
            start_sum = (np.array(self.pg_client2.get_queries_cost(self.workload)) * self.frequencies).sum()
            for index in self.candidates:
                oid = self.pg_client2.execute_create_hypo(index)
                cu_sum = (np.array(self.pg_client2.get_queries_cost(self.workload)) * self.frequencies).sum()
                x = (start_sum - cu_sum)/start_sum
                if x >= 0.5 and current_max < x:
                    current_max = x
                    current_index = index
                    current_index_len = current_index_len
                self.pg_client2.execute_delete_hypo(oid)
            if current_index is None or len(pre_is) >= self.max_count:
                break
            pre_is.append(current_index)
            self.pg_client2.execute_create_hypo(current_index)
        self.pre_create = pre_is
        self.pg_client2.delete_indexes()

        return pre_is

    def step(self, action):
        action = action[0]
        if self.currenct_index[action] != 0.0:
            # self.cost_trace_overall.append(self.last_cost_sum)
            # self.index_trace_overall.append(self.currenct_index)
            return self.last_state, 0, False

        self.index_oids[action] = self.pg_client1.execute_create_hypo(self.candidates[action])
        self.currenct_index[action] = 1.0
        oids : List[float] = list()
        oids.append(self.index_oids[action])
        storage_cost = self.pg_client1.get_storage_cost(oids)[0]
        # print(storage_cost)
        self.current_storage_sum += storage_cost
        self.current_index_storage[action] = storage_cost
        self.current_index_count += 1

        # reward & performance gain
        current_cost_info = np.array(self.pg_client1.get_queries_cost(self.workload))*self.frequencies
        current_cost_sum = current_cost_info.sum()
        if current_cost_sum < self.last_cost_sum:
            self.imp_count += 1
        # performance_gain_current = self.init_cost_sum - current_cost_sum
        # performance_gain_current = (self.last_cost_sum - current_cost_sum)/self.last_cost_sum
        # performance_gain_avg = performance_gain_current.round(1)
        # self.performance_gain[action] = performance_gain_avg
        # monitor info
        # self.cost_trace_overall.append(current_cost_sum)

        # update
        self.last_cost = current_cost_info
        self.last_state = np.append(self.frequencies, self.currenct_index)
        #self.last_state = self.currenct_index
        #deltac0 = (self.init_cost_sum - current_cost_sum) / self.init_cost_sum
        #deltac1 = (self.last_cost_sum - current_cost_sum) / self.last_cost_sum
        c = 1e-11
        reward = (self.last_cost_sum - current_cost_sum) / self.init_cost_sum - c
        '''if reward >= 0:
            reward -= c
        else:
            reward += c
        increase_flag = 1
        if self.last_cost_sum < current_cost_sum:
            increase_flag = -1
        reward = increase_flag * math.sqrt(math.fabs(self.last_cost_sum - current_cost_sum))'''
        # print(deltac0)
        '''deltac0 = max(0.000003, deltac0)
        if deltac0 == 0.000003:
            reward = -10
        else:
            reward = math.log(0.0003, deltac0)'''
        #print('reward is '+str(reward))
        '''deltac0 = self.init_cost_sum/current_cost_sum
        deltac1 = self.last_cost_sum/current_cost_sum
        reward = math.log(deltac0,10)'''
        self.last_cost_sum = current_cost_sum
        self.cost_trace_overall.append(current_cost_sum)

        self.index_trace_overall.append(self.currenct_index)
        if self.current_index_count >= self.max_count:

            return self.last_state, reward, True
        else:
            return self.last_state, reward, False
            # re5 return self.last_state, reward, False

    def reset(self, max_index_count):
        self.last_state = self.init_state
        self.last_cost = self.init_cost
        self.last_cost_sum = self.init_cost_sum
        # self.index_trace_overall.append(self.currenct_index)
        self.index_oids = np.zeros((len(self.candidates),), dtype=np.int)
        self.performance_gain = np.zeros((len(self.candidates),), dtype=np.float)
        self.current_index_count = 0
        self.current_min_cost = np.array(self.pg_client1.get_queries_cost(self.workload)).sum()
        self.current_min_index = np.zeros((len(self.candidates),), dtype=np.float)
        self.currenct_index = np.zeros((len(self.candidates),), dtype=np.float)
        self.current_index_storage = np.zeros((len(self.candidates),), dtype=np.float)
        self.pg_client1.delete_indexes()
        self.max_count = max_index_count
        # self.cost_trace_overall.append(self.last_cost_sum)
        '''if len(self.pre_create) > 0:  # 未加pre_create
            for i in self.pre_create:
                self.pg_client1.execute_create_hypo(i)
            self.init_cost_sum = (np.array(self.pg_client1.get_queries_cost(self.workload))*self.frequencies).sum()
            self.last_cost_sum = self.init_cost_sum'''
        self.last_state = np.append(self.frequencies, self.currenct_index)
        '''if len(self.pre_create) > 0:
            for i in self.pre_create:
                j = 0
                for index in self.candidates:
                    if i == index:
                        break
                    else: j += 1
                self.index_oids[j] = self.pg_client1.execute_create_hypo(i)
                self.currenct_index[j] = 1.0
                oids: List[float] = list()
                oids.append(self.index_oids[j])
                storage_cost = self.pg_client1.get_storage_cost(oids)[0]
                # print(storage_cost)
                self.current_storage_sum += storage_cost
                self.current_index_storage[j] = storage_cost
                self.current_index_count += 1
            #self.init_cost_sum = (np.array(self.pg_client1.get_queries_cost(self.workload))*self.frequencies).sum()
            #self.last_cost_sum = self.init_cost_sum
            self.last_cost_sum = (np.array(self.pg_client1.get_queries_cost(self.workload))*self.frequencies).sum()
            self.last_state = np.append(self.frequencies, self.currenct_index)'''
        return self.last_state
