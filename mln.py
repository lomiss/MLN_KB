# -*- coding: utf-8 -*-
import os
import sys
from pracmln.utils.project import PRACMLNConfig
from pracmln.utils import locs
from pracmln.utils.config import global_config_filename
from pracmln.mlnlearn import MLNLearn
from pracmln import MLN, Database, query


class TextArea(object):
    def __init__(self):
        self.buffer = []

    def write(self, *args, **kwargs):
        self.buffer.append(args)


class konwledge_modelling():
    def read_data(self, pre_content):
        content = []
        for i in pre_content:
            element = i.split('\n')
            element = [x.replace(':', '_') for x in element if x != '']
            for j in element[0:]:
                splited = j.split('(')
                content.append((element[0], splited[0] + '(' + splited[1].upper()))
        return content

    def read_formula(self, formula):
        formula = [' ' + x.replace(' or ', ' v ').replace(' and ', ' ^ ').replace(':', '') for x in formula]
        formula = ['0 ' + x for x in formula if 'Exists ' not in x]
        return formula

    def read_predicate(self, predicate):
        predicate_list = [x.split('(')[0] for x in predicate]
        predicate_list2 = [x.split('(')[1].replace(' ', '').lower() for x in predicate]
        predicate = []
        for i in zip(predicate_list, predicate_list2):
            predicate.append(i[0] + '(' + i[1])
        return predicate

    def model_config(self, predicate, formula, database, mln_path, db_path):  # mln_path,db_path 為string
        base_path = os.getcwd()
        mln = MLN(grammar='StandardGrammar', logic='FirstOrderLogic')
        for i in predicate:
            mln << i
        for i in formula:
            mln << i
        mln.write()
        mln.tofile(base_path + '\\mln_data\\' + mln_path)  # 把谓语数据储存成 mln_path.mln 档案
        db = Database(mln)
        try:
            for i in enumerate(database):
                db << i[1][1]
        except:
            for j in database[i[0]::]:
                db << j[1]
        db.write()
        db.tofile(base_path + '\\mln_data\\' + db_path)  # 把证据数据储存成 db_path.db 档案
        return (db, mln)

    def activate_model(self, database, mln):
        DEFAULT_CONFIG = os.path.join(locs.user_data, global_config_filename)
        conf = PRACMLNConfig(DEFAULT_CONFIG)
        config = {}
        config['verbose'] = False
        config['discr_preds'] = 0
        config['db'] = database
        config['mln'] = mln
        config['ignore_zero_weight_formulas'] = True  # 0
        config['ignore_unknown_preds'] = True  # 0
        config['incremental'] = 0  # 0
        config['grammar'] = 'StandardGrammar'
        config['logic'] = 'FirstOrderLogic'
        config['method'] = 'BPLL'  # BPLL
        config['multicore'] = True
        config['profile'] = 0
        config['shuffle'] = 0
        config['prior_mean'] = 0
        config['prior_stdev'] = 10  # 5
        config['save'] = False
        config['use_initial_weights'] = 0
        config['use_prior'] = 0
        config['infoInterval'] = 500
        config['resultsInterval'] = 1000
        conf.update(config)
        stdout = sys.stdout
        sys.stdout = TextArea()
        print('training...')
        learn = MLNLearn(conf, mln=mln, db=database)
        learn.run()
        print('finished...')
        text_area, sys.stdout = sys.stdout, stdout
        return text_area.buffer

    def inference(self, query_list, data, mln, rate):  # 推理查询未知的命题
        ans = {}
        for i in query_list:
            tmp = dict(query(queries=i, method='MC-SAT', mln=mln, db=data, verbose=False, multicore=True).run().results)
            # ans.update(tmp)
            for k, v in tmp.items():
                if v >= rate:
                    ans[k] = v
        return ans


# if __name__ == '__main__':
#     predicate = ['rich(n)', 'cheap(n)', 'supply(a)', 'adapt(n, a)', 'cold(n)']
#     formula = ['rich(p1) => cheap(p2)', 'adapt(p1, p2) => rich(p1)',
#                'cold(p1) ^ !adapt(p2, p3) => !supply(p4)']
#     pre_content = ['rich(corn)', 'cheap(price)', 'supply(many)', 'adapt(corn, strong)', 'cold(weather)']
#     query_list = ['rich(a)']
#     konwledge = konwledge_modelling()
#     predicate = konwledge.read_predicate(predicate)
#     formula = konwledge.read_formula(formula)
#     database = konwledge.read_data(pre_content)
#     data, mln = konwledge.model_config(predicate, formula, database, 'markov.mln', 'markov.db')
#     output = konwledge.activate_model(data, mln)
#     print(output)
#     ans = konwledge.inference(query_list, data, mln, 0)
#     for each in ans:
#         print(each)
