import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import numpy as np
import pickle

import Model.ModelDQNFixCount as model
import Model.Model3DQNFixStorage as model2
import Model.ModelDuelingDQNFixCount as duelingdqn
import Model.ModelActorCritic as acmodel
import Model.MADQNFixCountOnDuelingPR as madqnPR
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json

# 动态负载下索引推荐测试入口-动态负载下的第一个实验，不包括索引更新的场景

def One_Run_DQN(is_fixcount, conf, __x, freq, workload, index_candidates, query2cands, dicts):

    if is_fixcount:
        # agent = model.DQN(workload[:], index_candidates, 'hypo', conf,  freq)
        # agent = duelingdqn.DQN(workload[:], index_candidates, 'hypo', conf,  freq, True, True, True)
        agent = madqnPR.DQN(workload[:], index_candidates, 'hypo', conf, freq, True, False, False, query2cands, dicts)
        #agent = acmodel.AC(workload[:], index_candidates, 'hypo', conf, freq)
        current_best_cost = agent.train(True, __x)
        return current_best_cost
    else:
        agent = model2.DQN(workload, index_candidates, 'hypo', conf)
        _indexes, storages = agent.train(False, __x)
        indexes = []
        for _i, _idx in enumerate(_indexes):
            if _idx == 1.0:
                indexes.append(index_candidates[_i])
        return indexes


conf21 = {'LR': 0.003, 'EPISILO': 0.03, 'Q_ITERATION': 200, 'U_ITERATION': 5, 'BATCH_SIZE': 32, 'GAMMA': 0.95,
          'EPISODES': 1000, 'LEARNING_START': 1000, 'MEMORY_CAPACITY': 20000, 'NAME': 'DDQN-1', 'Q_im': 1e-8}

conf = {'LR': 0.1, 'EPISILO': 0.1, 'Q_ITERATION': 9, 'U_ITERATION': 3, 'BATCH_SIZE': 8, 'GAMMA': 0.9,
        'EPISODES': 800, 'LEARNING_START': 400, 'MEMORY_CAPACITY': 800, 'NAME': 'MA'}


# is_fixcount == True, constraint is the index number
# is_fixcount == False, constraint is the index storage unit
def entry(is_fixcount, constraint):
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

    f = open("querys2candidates", 'r')
    query2cands = eval(f.read())
    f.close()
    print("=========load query2candidates=========")

    print('=====load distinction of candidates=====')
    file = open('candidates2distinct', 'r')
    js = file.read()
    dicts = json.loads(js)
    current_best_costs = []
    if is_fixcount:
        '''for i in range(1):
            freq = test_ds[1]
            current_best_cost = One_Run_DQN(is_fixcount, conf21, constraint,  freq, workload, index_candidates, query2cands, dicts)
            current_best_costs.append(current_best_cost)
            print(str(i) + "test done")'''
        for i in range(len(test_ds)):
            current_best_cost = 0
            for j in range(5):
                freq = test_ds[i]
                current_best_cost += One_Run_DQN(is_fixcount, conf21, constraint, freq, workload, index_candidates,
                                                query2cands, dicts)
            current_best_cost /= 5
            print("current best cost_sum is:" + str(current_best_cost))
            current_best_costs.append(current_best_cost)
            print(str(i) + "test done")
            if i == 120:
                break
    else:
        '''for i in range(len(test_ds)):
            freq = test_ds[i]
            One_Run_DQN(is_fixcount, conf, constraint, freq, workload, index_candidates, query2cands)'''
    '''plt.figure()
    x = range(len(current_best_costs))
    y = np.array(current_best_costs)
    plt.title(conf21['NAME'])
    plt.xlabel("workload")
    plt.ylabel("Cost(W,I)")
    plt.plot(x, y, marker='x')
    plt.savefig("madqn_cost_train.png", dpi=120)
    plt.clf()
    plt.close()'''
    '''cost_avg = 0
    for i in range(len(current_best_costs)):
        cost_avg += current_best_costs[i]
    cost_avg = cost_avg / len(current_best_costs)
    print("the best cost of MADQN-TRUE is:"+str(cost_avg)+"  after "+str(len(current_best_costs))+" times")'''
    # 保存训练结果
    '''train_ds_costs = 'train_ds_costs_DuelingDQN-on-PRDQN'
    with open(train_ds_costs, "a+") as fp:
        json.dump(current_best_costs, fp)'''
    #保存测试数据
    test_ds_costs = 'test_ds_costs_DDQN-1'
    with open(test_ds_costs, "a+") as fp:
        json.dump(current_best_costs, fp)

entry(True, 5)
