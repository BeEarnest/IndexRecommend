import pickle
import psqlparse
import sys
import os
curPath=os.path.abspath(os.path.dirname(__file__))
rootPath=os.path.split(curPath)[0]
sys.path.append(rootPath)

import Preprocess
from Preprocess import Dataset as ds
from Utility import Encoding as en
from Utility import ParserForIndex as pi
import codecs
import json
import Utility.PostgreSQL as pg

enc = en.encoding_schema()
# path to your tpch_directory/dbgen
work_dir = "/home/fj/2.18.0_rc2/dbgen"
w_size = 14
wd_generator = ds.TPCH(work_dir, w_size)
workload = wd_generator.gen_workloads()

parser = pi.Parser(enc['attr'])
directory = '../Entry/'

def gen_distinction(col):
    # 解析colname
    table_name, col_name = col.split('#')
    pg_client = pg.PGHypo()
    distinct = pg_client.get_distinction(table_name, col_name)
    return distinct

# 为所有候选索引列生成区分度，并保存在字典中
def gen_dict_for_candidates(file):
    print('=====load candidate =====')
    cf = open(file, 'rb')
    index_candidates = pickle.load(cf)
    # 定义字典
    d = {}
    # 构建字典
    i = 0
    for index in index_candidates:
        distinct = gen_distinction(index)
        d[index] = distinct
        print("save"+str(i)+" distinct")
        i += 1
    # 将字典保存至文件中
    file = directory+'candidates2distinct'
    with codecs.open(file, 'w', encoding='utf-8') as f:
        json.dump(d, f)


# 为workload中的所有query生成candidates
def gen_i():
    #added_i = set()
    f_i = set()
    # 定义字典保存每条query对应的候选索引集
    d = {}
    for i in range(len(workload)):
        #print(workload[i])
        b = psqlparse.parse_dict(workload[i])
        parser.parse_stmt(b[0])
        parser.gain_candidates()
        '''if i == 8:
            added_i.add('lineitem#l_shipmode')
            added_i.add('lineitem#l_orderkey,l_shipmode')
            added_i.add('lineitem#l_shipmode,l_orderkey')'''
        #f_i = parser.index_candidates | added_i
        f_i = parser.index_candidates | f_i
        d[i] = parser.index_candidates
    f_i = list(f_i)
    f_i.sort()
    with open(directory+'cands'+str(w_size)+'.pickle', 'wb') as df:
        pickle.dump(list(f_i), df, protocol=0)
    print("save new candidates successfully.")

    gen_dict_for_candidates(directory+'cands'+str(w_size)+'.pickle')
    print("save distinct successfully.")

    f = open(directory+'querys2candidates', 'w')
    f.write(str(d))
    f.close()
    print("save dict querys2candidates successfully.")

    # 写入workload
    with open(directory+'workload'+str(w_size)+'.pickle', 'wb') as df:
        pickle.dump(workload, df, protocol=0)
    print("save new workload successfully.")

gen_i()
