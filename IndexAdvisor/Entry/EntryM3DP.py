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
import Model.MADQNFixCountOnDuelingPR as madqnPR
import Model.ModelActorCritic as acmodel
import matplotlib.pyplot as plt
import json

# 静态负载下索引推荐测试入口

def One_Run_DQN(is_fixcount, conf, __x, freq, workload, index_candidates, query2cands, dicts):

    if is_fixcount:
        # 原始DQN
        # agent = model.DQN(workload[:], index_candidates, 'hypo', conf,  freq)
        # 基于原始DQN的Dueling DQN
        # agent = duelingdqn.DQN(workload[:], index_candidates, 'hypo', conf,  freq, True, True, True)
        # MADQN模型，is_double, is_ps, is_dnn取不同值对应不同的模型
        # False,False,False代表基于Nature DQN扩展原始DQN
        # True,False,False代表基于DDQN扩展原始DQN
        # True,True,False代表基于PR DQN扩展原始DQN
        # True,True,True代表基于PR DQN+Dueling扩展原始DQN
        agent = madqnPR.DQN(workload[:], index_candidates, 'hypo', conf, freq, True, False, False, query2cands, dicts)
        # AC算法，但该模型无法收敛
        # agent = acmodel.AC(workload[:], index_candidates, 'hypo', conf,  True,  freq)
        # 返回平均时间步
        avg_time_step = agent.train(True, __x)
        return avg_time_step
        # 返回当前最优查询代价
        # current_best_cost = agent.train(True, __x)
        # return current_best_cost
        # 返回当前最优查询代价下的最优索引
        # current_best_index = agent.train(True, __x)
        # return current_best_index
    else:  # 约束条件为索引可占用的最大存储空间，该功能未做修改和测试
        agent = model2.DQN(workload, index_candidates, 'hypo', conf)
        _indexes, storages = agent.train(False, __x)
        indexes = []
        for _i, _idx in enumerate(_indexes):
            if _idx == 1.0:
                indexes.append(index_candidates[_i])
        return indexes

# 配置参数
conf21 = {'LR': 0.003, 'EPISILO': 0.03, 'Q_ITERATION': 1000, 'U_ITERATION': 5, 'BATCH_SIZE': 32, 'GAMMA': 0.95,
          'EPISODES': 1000, 'LEARNING_START': 1000, 'MEMORY_CAPACITY': 20000, 'NAME': 'Entropy-ON-DRL', 'Q_im': 1e-4}

conf = {'LR': 0.1, 'EPISILO': 0.1, 'Q_ITERATION': 9, 'U_ITERATION': 3, 'BATCH_SIZE': 8, 'GAMMA': 0.9,
        'EPISODES': 800, 'LEARNING_START': 400, 'MEMORY_CAPACITY': 800, 'NAME': 'MA'}


# is_fixcount == True, constraint is the index number
# is_fixcount == False, constraint is the index storage unit

def entry(is_fixcount, constraint):
    # 1 导入训练数据
    # 1.1 导入负载的查询分布
    print('=====load train data=====')
    file = open('workload_distribution_train', 'r')
    js = file.read()
    train_ds = json.loads(js)
    # 1.2 导入查询负载
    print('=====load workload=====')
    wf = open('workload14.pickle', 'rb')
    workload = pickle.load(wf)
    # 1.2 导入候选索引集
    print('=====load candidate =====')
    cf = open('cands14.pickle', 'rb')
    index_candidates = pickle.load(cf)
    # 1.4 导入预先生成的负载中的每条查询语句与其候选索引之间的映射关系
    print("=========load query2candidates=========")
    f = open("querys2candidates", 'r')
    query2cands = eval(f.read())
    f.close()
    # 1.5 导入预先生成的候选索引集中每个索引的区分度
    print('=====load distinction of candidates=====')
    file = open('candidates2distinct', 'r')
    js = file.read()
    dicts = json.loads(js)
    current_best_costs = []
    current_best_cost = 0
    avg_time_step = 0
    if is_fixcount:
        # 针对负载中的一种固定的查询分布，循环10次调用模型，得到平均最优的查询代价current_best_cost
        '''for i in range(10):
            freq = train_ds[0]
            current_best_cost += One_Run_DQN(is_fixcount, conf21, constraint,  freq, workload, index_candidates, query2cands, dicts)
            print(str(i) + "train done")
        current_best_cost /= 10
        print("current best cost_sum is:" + str(current_best_cost))'''

        # 针对负载中的一种固定的查询分布，循环10次调用模型，得到平均最优的查询时间步avg_time_step
        for i in range(10):
            freq = train_ds[0]
            avg_time_step += One_Run_DQN(is_fixcount, conf21, constraint,  freq, workload, index_candidates, query2cands, dicts)
            print(str(i) + "train done")
        avg_time_step /= 10
        print("current time step is:" + str(avg_time_step))

        # 单次运行模型，以获取模型的收敛情况
        '''freq = train_ds[0]
        current_best_index = One_Run_DQN(is_fixcount, conf21, constraint, freq, workload, index_candidates, query2cands,
                                     dicts)
        print(current_best_index)'''

    else:
        for i in range(len(train_ds)):
            freq = train_ds[i]
            One_Run_DQN(is_fixcount, conf, constraint, freq, workload, index_candidates, query2cands)

entry(True, 5)
