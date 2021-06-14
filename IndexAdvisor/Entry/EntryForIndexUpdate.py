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
import Model.MADQNFixCountOnDuelingPR as madqnPR
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json

# 动态负载下索引推荐测试入口-动态负载下的第二个实验，包括索引更新的场景

conf21 = {'LR': 0.003, 'EPISILO': 0.03, 'Q_ITERATION': 200, 'U_ITERATION': 5, 'BATCH_SIZE': 32, 'GAMMA': 0.95,
          'EPISODES': 1000, 'LEARNING_START': 1000, 'MEMORY_CAPACITY': 20000, 'NAME': 'MADDQN', 'Q_im': 1e-8}

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
    before_best_costs = []
    latest_best_cost = 0
    current_best_cost = 0
    current_best_index = None
    constraints = []
    if is_fixcount:
        for i in range(len(test_ds)):
            freq = test_ds[i]
            agent = madqnPR.DQN(workload[:], index_candidates, 'hypo', conf21, freq, True, False, False, query2cands,
                                dicts)
            latest_best_cost = agent.get_latest_best_cost(current_best_index) #获取当前索引配置下的cost
            if current_best_cost == 0 and current_best_index is None:  # 初始，第一次不存
                # before_best_costs.append(latest_best_cost)
                current_best_index = agent.train(True, constraint)
                current_best_cost = agent.get_latest_best_cost(current_best_index)
                #current_best_costs.append(current_best_cost)
                #perform_gain = (latest_best_cost - current_best_cost) / latest_best_cost
                #currrent_perform_gains.append(perform_gain)
            else:  # current_best_cost是上一次的，latest_best_cost是当前负载下的
                perform_gain = (current_best_cost - latest_best_cost)/current_best_cost
                before_best_costs.append(latest_best_cost)  # 新的负载在当前索引配置下的cost
                if perform_gain < -0.6:  # 性能下降超过阈值0.8
                    '''_current_best_index = agent.train(True, constraint)
                    _latest_best_cost = agent.get_latest_best_cost(_current_best_index)
                    _perform_gain = (current_best_cost - _latest_best_cost) / current_best_cost'''
                    _current_best_index = None
                    _latest_best_cost = 0
                    if perform_gain < -0.6:  # constraints不变,模型重新训练一次
                        _current_best_index = agent.train(True, constraint)
                        _latest_best_cost = agent.get_latest_best_cost(_current_best_index)
                        perform_gain = (current_best_cost - _latest_best_cost)/current_best_cost
                    if perform_gain < -0.6:  # 性能下降仍然超过0.8
                        _constraint = constraint
                        while perform_gain < -0.6 and _constraint < 10:  # 增大索引数来提升性能，最大索引数不得超过10
                            _constraint += 1
                            _current_best_index = agent.train(True, _constraint)
                            _latest_best_cost = agent.get_latest_best_cost(_current_best_index)
                            perform_gain = (current_best_cost - _latest_best_cost) / current_best_cost
                        if perform_gain >= -0.6:  # 说明在最大索引数内性能得到了提升
                            current_best_index = _current_best_index
                            current_best_cost = _latest_best_cost
                            current_best_costs.append(current_best_cost)
                            constraint = _constraint
                        else:  # 最大索引数下仍无性能提升，保持现状
                            current_best_cost = latest_best_cost  # 更新当前最优的查询代价
                            current_best_costs.append(current_best_cost)

                    else:
                        current_best_index = _current_best_index
                        current_best_cost = _latest_best_cost
                        current_best_costs.append(current_best_cost)
                else:
                    current_best_cost = latest_best_cost
                    current_best_costs.append(current_best_cost)
            print(str(i) + "test done")
            print("constraint is: "+str(constraint))
            if i != 0:
                constraints.append(constraint)
            if i == 120:
                break
    #保存测试数据
    test_ds_before_costs_update = 'test_ds_before_costs_MADDQN_UPDATE'
    with open(test_ds_before_costs_update, "a+") as fp:
        json.dump(before_best_costs, fp)
    test_ds_current_costs_update = 'test_ds_current_costs_MADDQN_UPDATE'
    with open(test_ds_current_costs_update, "a+") as fp:
        json.dump(current_best_costs, fp)
    test_ds_constraints_update = 'test_ds_constraints_MADDQN_UPDATE'
    with open(test_ds_constraints_update, "a+") as fp:
        json.dump(constraints, fp)

entry(True, 5)
