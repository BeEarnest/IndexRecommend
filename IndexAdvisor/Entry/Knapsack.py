#coding:utf-8
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import pickle
import json
import numpy as np
from Utility import PostgreSQL as pg

# 基于背包算法实现的索引推荐
class Knapsack:
    def __init__(self, workload, candidates, freq):
        self.workload = workload
        self.candidates = candidates
        self.pg_client = pg.PGHypo()
        self.freq = freq
        self.init_cost = np.array(self.pg_client.get_queries_cost(workload)) * self.freq
        self.init_cost_sum = self.init_cost.sum()  # 未建任何索引
        self.last_cost = self.init_cost
        self.last_cost_sum = self.init_cost_sum
        self.v = []
        for index in self.candidates:
            oid = self.pg_client.execute_create_hypo(index)
            cu_sum = (np.array(self.pg_client.get_queries_cost(self.workload)) * self.freq).sum()
            c = 1e-11
            reward = (self.last_cost_sum - cu_sum) / self.init_cost_sum - c
            self.v.append(reward)
            self.pg_client.execute_delete_hypo(oid)

    def bag(self, n, c, w):
        """
        测试数据：
        n = 6  物品的数量，
        c = 10 书包能承受的重量，
        w = [2, 2, 3, 1, 5, 2] 每个物品的重量，
        v = [2, 3, 1, 5, 4, 3] 每个物品的价值
        """
        # 置零，表示初始状态
        value = [[0 for j in range(c + 1)] for i in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, c + 1):
                value[i][j] = value[i - 1][j]
                # 背包总容量够放当前物体，遍历前一个状态考虑是否置换
                if j >= w[i - 1] and value[i][j] < value[i - 1][j - w[i - 1]] + self.v[i - 1]:
                    value[i][j] = value[i - 1][j - w[i - 1]] + self.v[i - 1]
                    #self.getValueForIndex(i-1) # 更新v
        # for x in value:
        #    print(x)
        return value

    def show(self, n, c, w, value):
        #print('最大价值为:', value[n][c])
        x = [0 for i in range(n)]
        j = c
        for i in range(n, 0, -1):
            if value[i][j] > value[i - 1][j]:
                x[i - 1] = 1
                j -= w[i - 1]
        '''print('背包中所装物品为:')
        for i in range(n):
            if x[i] == 1:
                print('第', i+1, '个,', end='')'''
        return x

    #def getValueForIndex(self, current_index):




if __name__ == '__main__':
    '''print('=====load train data=====')
    file = open('workload_distribution_train', 'r')
    js = file.read()
    train_ds = json.loads(js)'''
    print('=====load test data=====')
    file = open('workload_distribution_test', 'r')
    js = file.read()
    test_ds = json.loads(js)
    print('=====load workload=====')
    wf = open('workload14.pickle', 'rb')
    workload = pickle.load(wf)
    print('=====load candidate =====')
    cf = open('cands14.pickle', 'rb')
    index_candidates = pickle.load(cf)

    current_best_costs = []
    n = len(index_candidates)
    c = 5
    w = [1 for j in range(n)]
    for i in range(len(test_ds)):
        # freq = train_ds[0]
        freq = test_ds[i]
        freq = np.array(freq) / np.array(freq).sum()
        knapsack = Knapsack(workload, index_candidates, freq)
        value = knapsack.bag(n, c, w)
        x = knapsack.show(n, c, w, value)
        knapsack.pg_client.delete_indexes()
        indexes = []
        for _i, _idx in enumerate(x):
            if _idx == 1.0:
                indexes.append(knapsack.candidates[_i])
        for f_index in indexes:
            knapsack.pg_client.execute_create_hypo(f_index)
        current_best_cost_sum = (
                np.array(knapsack.pg_client.get_queries_cost(knapsack.workload)) * knapsack.freq).sum()
        print("current best cost_sum is:" + str(current_best_cost_sum))
        current_best_costs.append(current_best_cost_sum)
        print(str(i) + "test done")
        if i == 120:
            break
    #保存测试数据
    test_ds_costs = 'test_ds_costs_Knapsack'
    with open(test_ds_costs, "a+") as fp:
        json.dump(current_best_costs, fp)
