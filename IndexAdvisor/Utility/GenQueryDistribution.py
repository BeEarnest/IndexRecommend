import numpy as np
import os
import json
import random

directory = '../Entry/'

def split(full_list,shuffle=False,ratio=0.2):
  n_total = len(full_list)
  offset = int(n_total * ratio)
  if n_total==0 or offset<1:
    return [],full_list
  if shuffle:
    random.shuffle(full_list)
  sublist_1 = full_list[:offset]
  sublist_2 = full_list[offset:]
  return sublist_1,sublist_2

#生成W_NUM个w_size大小查询负载的查询分布
def gen_query_distribution(w_num, w_size):
    all = []
    for i in range(w_num):
        _freq = []
        for j in range(w_size):
            f = np.random.random_integers(1000)
            f = int(f)
            _freq.append(f)
        all.append(_freq)
    train_ds = directory + 'workload_distribution_train'
    test_ds = directory + 'workload_distribution_test'
    sublist_1, sublist_2 = split(all, shuffle=True, ratio=0.7)
    if os.path.isfile(train_ds):
        os.remove(train_ds)
    with open(train_ds, "w") as fp:
        json.dump(sublist_1, fp)
    if os.path.isfile(test_ds):
        os.remove(test_ds)
    with open(test_ds, "w") as fp:
        json.dump(sublist_2, fp)

gen_query_distribution(1000, 14)