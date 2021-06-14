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


enc = en.encoding_schema()
# path to your tpch_directory/dbgen
work_dir = "/home/fj/2.18.0_rc2/dbgen"
w_size = 14
wd_generator = ds.TPCH(work_dir, w_size)
workload = wd_generator.gen_workloads()

parser = pi.Parser(enc['attr'])

# 生成候选索引
def gen_i(__x):
    added_i = set()
    for i in range(len(workload)):
        if i > __x:
            continue
        #print(workload[i])
        b = psqlparse.parse_dict(workload[i])
        parser.parse_stmt(b[0])
        parser.gain_candidates()
        if i == 8:
            added_i.add('lineitem#l_shipmode')
            added_i.add('lineitem#l_orderkey,l_shipmode')
            added_i.add('lineitem#l_shipmode,l_orderkey')
    f_i = parser.index_candidates | added_i
    f_i = list(f_i)
    f_i.sort()
    with open('cands'+str(__x+1)+'.pickle', 'wb') as df:
        pickle.dump(list(f_i), df, protocol=0)


for i in range(0, 14):
    gen_i(i)
