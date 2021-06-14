import os
from configparser import ConfigParser
from typing import List
import psycopg2 as pg
import pandas as pd
import time

# 连接PG数据库
class PGHypo:
    def __init__(self):
        config_raw = ConfigParser()
        config_raw.read(os.path.abspath('..') + '/configure.ini')
        defaults = config_raw.defaults()
        self.host = defaults.get('pg_ip')
        self.port = defaults.get('pg_port')
        self.user = defaults.get('pg_user')
        self.password = defaults.get('pg_password')
        self.database = defaults.get('pg_database')
        self.conn = pg.connect(database=self.database, user=self.user, password=self.password, host=self.host,
                                     port=self.port)

    def close(self):
        self.conn.close()

    def execute_create_hypo(self, index):
        schema = index.split("#")
        sql = "SELECT indexrelid FROM hypopg_create_index('CREATE INDEX ON " + schema[0] + "(" + schema[1] + ")') ;"
        cur = self.conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return int(rows[0][0])

    def execute_delete_hypo(self, oid):
        sql = "select * from hypopg_drop_index(" + str(oid) + ");"
        cur = self.conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        flag = str(rows[0][0])
        if flag == "t":
            return True
        return False

    def get_queries_cost(self, query_list):
        cost_list: List[float] = list()
        cur = self.conn.cursor()
        for i, query in enumerate(query_list):
            query = "explain " + query
            cur.execute(query)
            rows = cur.fetchall()
            df = pd.DataFrame(rows)
            cost_info = str(df[0][0])
            cost_list.append(float(cost_info[cost_info.index("..") + 2:cost_info.index(" rows=")]))
        return cost_list

    def get_storage_cost(self, oid_list):
        costs = list()
        cur = self.conn.cursor()
        for i, oid in enumerate(oid_list):
            if oid == 0:
                continue
            sql = "select * from hypopg_relation_size(" + str(oid) +");"
            cur.execute(sql)
            rows = cur.fetchall()
            df = pd.DataFrame(rows)
            cost_info = str(df[0][0])
            cost_long = int(cost_info)
            costs.append(cost_long)
            # print(cost_long)
        return costs

    def execute_sql(self, sql):
        cur = self.conn.cursor()
        cur.execute(sql)
        self.conn.commit()

    def get_distinction(self, table_name, col_name):
        sql1 = "select count(*) from "+table_name
        cur = self.conn.cursor()
        cur.execute(sql1)
        rows = cur.fetchall()
        sql2 = "select count(*) from (select distinct "+col_name+" from "+table_name+") as b"
        cur = self.conn.cursor()
        cur.execute(sql2)
        rows2 = cur.fetchall()
        return float(rows2[0][0])/float(rows[0][0])

    def delete_indexes(self):
        sql = 'select * from hypopg_reset();'
        self.execute_sql(sql)

    def get_sel(self, table_name, condition):
        cur = self.conn.cursor()
        totalQuery = "select * from " + table_name + ";"
        cur.execute("EXPLAIN " + totalQuery)
        rows = cur.fetchall()[0][0]
        #     print(rows)
        #     print(rows)
        total_rows = int(rows.split("rows=")[-1].split(" ")[0])

        resQuery = "select * from " + table_name + " Where " + condition + ";"
        # print(resQuery)
        cur.execute("EXPLAIN  " + resQuery)
        rows = cur.fetchall()[0][0]
        #     print(rows)
        select_rows = int(rows.split("rows=")[-1].split(" ")[0])
        return select_rows/total_rows

    def get_rel_cost(self, query_list):
        print("real")
        cost_list: List[float] = list()
        cur = self.conn.cursor()
        for i, query in enumerate(query_list):
            _start = time.time()
            query = "explain analyse" + query
            cur.execute(query)
            _end = time.time()
            cost_list.append(_end-_start)
        return cost_list

    def create_indexes(self, indexes):
        i = 0
        for index in indexes:
            schema = index.split("#")
            sql = 'CREATE INDEX START_X_IDx' + str(i) + ' ON ' + schema[0] + "(" + schema[1] + ');'
            print(sql)
            self.execute_sql(sql)
            i += 1

    def delete_t_indexes(self):
        sql = "SELECT relname from pg_class where relkind = 'i' and relname like 'start_x_idx%';"
        print(sql)
        cur = self.conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        indexes = []
        for row in rows:
            indexes.append(row[0])
        print(indexes)
        for index in indexes:
            sql = 'drop index ' + index + ';'
            print(sql)
            self.execute_sql(sql)

    def get_tables(self, schema):
        tables_sql = 'select tablename from pg_tables where schemaname=\''+schema+'\';'
        cur = self.conn.cursor()
        cur.execute(tables_sql)
        rows = cur.fetchall()
        table_names = list()
        for i, table_name in enumerate(rows):
            table_names.append(table_name[0])
        return table_names

    def get_attributes(self, table_name, schema):
        attrs_sql = 'select column_name, data_type from information_schema.columns where table_schema=\''+schema+'\' and table_name=\''+table_name+'\''
        cur = self.conn.cursor()
        cur.execute(attrs_sql)
        rows = cur.fetchall()
        attrs = list()
        for i, attr in enumerate(rows):
            info = str(attr[0]) + "#" + str(attr[1])
            attrs.append(info)
        return attrs
