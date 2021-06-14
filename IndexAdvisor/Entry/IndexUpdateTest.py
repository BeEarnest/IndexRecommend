import pickle
import json
import Model.MADQNFixCountOnDuelingPR as madqnPR
import Model.Model3DQNFixStorage as model2
from Utility import PostgreSQL as pg
import numpy as np

def One_Run_DQN(is_fixcount, conf, __x, freq, workload, index_candidates, query2cands, dicts):

    if is_fixcount:
        #agent = model.DQN(workload[:], index_candidates, 'hypo', conf,  freq)
        # agent = model3.UpdatedDQN(workload[:], index_candidates, 'hypo', conf, False, False, False,  freq, query2cands)
        # agent = natureDQN.DQN(workload[:], index_candidates, 'hypo', conf,  freq)
        # agent = ddqn.DQN(workload[:], index_candidates, 'hypo', conf,  freq, True)
        # agent = prdqn.DQN(workload[:], index_candidates, 'hypo', conf,  freq, True, True)
        # agent = duelingdqn.DQN(workload[:], index_candidates, 'hypo', conf,  freq, True, True, True)
        agent = madqnPR.DQN(workload[:], index_candidates, 'hypo', conf21, freq, True, False, False, query2cands,
                            dicts)
        '''before_cost_sum = agent.envx.last_cost_sum
        print('cost is :'+str(before_cost_sum)+' when has no index')'''
        current_best_index = agent.train(True, __x)
        return current_best_index
        '''current_best_cost = agent.train(True, __x)
        return current_best_cost'''
    else:
        agent = model2.DQN(workload, index_candidates, 'hypo', conf)
        _indexes, storages = agent.train(False, __x)
        indexes = []
        for _i, _idx in enumerate(_indexes):
            if _idx == 1.0:
                indexes.append(index_candidates[_i])
        return indexes


conf21 = {'LR': 0.003, 'EPISILO': 0.03, 'Q_ITERATION': 200, 'U_ITERATION': 5, 'BATCH_SIZE': 32, 'GAMMA': 0.95,
          'EPISODES': 1000, 'LEARNING_START': 1000, 'MEMORY_CAPACITY': 20000, 'NAME': 'MADDQN', 'Q_im': 1e-8}

conf = {'LR': 0.1, 'EPISILO': 0.1, 'Q_ITERATION': 9, 'U_ITERATION': 3, 'BATCH_SIZE': 8, 'GAMMA': 0.9,
        'EPISODES': 800, 'LEARNING_START': 400, 'MEMORY_CAPACITY': 800, 'NAME': 'MA'}

def test(is_fixcount, constraint):
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
    #freq = [247, 32, 868, 107, 632, 149, 48, 265, 509, 774, 674, 925, 152, 552]
    #freq1 = [1000, 10, 8, 5, 5, 7, 3, 1, 5, 7, 8, 10, 11, 4]
    #freq2 = [1000, 10, 8, 5, 5, 7, 3, 1, 5, 7, 8, 10, 11, 4]

    frequencies = [0.01886792, 0.18867925, 0.01886792, 0.09433962, 0.09433962, 0.01886792, 0.05660377, 0.01886792, 0.09433962, 0.13207547, 0.03773585, 0.18867925, 0.01886792, 0.01886792]
    print('freq is: '+str(frequencies))
    #current_best_index = [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    '''current_best_index = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pg_client = pg.PGHypo()
    indexes = []
    for _i, _idx in enumerate(current_best_index):
        if _idx == 1.0:
            indexes.append(index_candidates[_i])
    for f_index in indexes:
        pg_client.execute_create_hypo(f_index)

    current_cost_info = np.array(pg_client.get_queries_cost(workload)) * frequencies
    current_cost_sum = current_cost_info.sum()
    print(current_cost_sum)
    print("performence decrease: " + str((current_cost_sum - 241505.27074723248) / 241505.27074723248))'''

    current_best_index = One_Run_DQN(is_fixcount, conf21, constraint, frequencies, workload, index_candidates, query2cands, dicts)
    pg_client = pg.PGHypo()
    indexes = []
    for _i, _idx in enumerate(current_best_index):
        if _idx == 1.0:
            indexes.append(index_candidates[_i])
    for f_index in indexes:
        pg_client.execute_create_hypo(f_index)

    current_cost_info = np.array(pg_client.get_queries_cost(workload)) * frequencies
    current_cost_sum = current_cost_info.sum()
    print(current_cost_sum)
    print(indexes)

test(True,5)