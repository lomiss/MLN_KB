# -*- coding: utf-8 -*-
import os
import triple_extraction
import causality_extract
import complex_extract
import collection_mi
from pyltp import Postagger, Segmentor
from itertools import chain
from datetime import datetime

# 全局变量
file_name = 'resource_cleaned/maize.txt'
markov_formula = 'mln_data/markov_formula.txt'
markov_predicate = 'mln_data/markov_predicate.txt'
markov_text_num = 'mln_data/markov_text_num.txt'
markov_num_text = 'mln_data/markov_num_text.txt'

formula_list = set()
predicate_list = {}

text_num = {}
num_text = {}
sentence = []
triple = []

LTP_DIR = "ltp_data_v3.4.0/"
extractor = triple_extraction.TripleExtractor()
postagger = Postagger()
segmentor = Segmentor()
postagger.load(os.path.join(LTP_DIR, "pos.model"))
segmentor.load(os.path.join(LTP_DIR, "cws.model"))


# 读玉米语料处理三元组
def triple_text(file_name, article, second):
    global triple
    start = datetime.now()
    i = 0
    with open(file_name, 'r', encoding='utf-8') as file_object:
        for each_article in file_object:
            if i == article:
                return triple
            print(i)
            end = datetime.now()
            if (end - start).seconds > second:
                return []
            sentence_svos = extractor.triples_main(each_article)
            for each in sentence_svos:
                triple.append(sentence_svos[each])
            i += 1
    return triple


# 将文本字典化并返回sentence
def dict_text():
    cnt = 1
    for each in triple:
        if not len(each):
            continue
        each = str(each)
        tmp = each[2:len(each) - 2].split('], [')
        logic_sentence = []
        for each_triple in tmp:
            print(each_triple)
            tmp1 = each_triple.split(', ')
            a = tmp1[0][1:len(tmp1[0]) - 1]
            b = tmp1[1][1:len(tmp1[1]) - 1]
            if a not in text_num.keys():
                text_num[a] = cnt
                num_text[str(cnt)] = a
                tmpp = cnt
                cnt += 1
            else:
                text_num[a + '_' + str(cnt)] = cnt
                num_text[str(cnt)] = a + '_' + str(cnt)
                tmpp = cnt
                cnt += 1
            if b not in text_num.keys():
                text_num[b] = cnt
                num_text[str(cnt)] = b
                cnt += 1
            if len(tmp1) == 3:
                c = tmp1[2][1:len(tmp1[2]) - 1]
                if c not in text_num.keys():
                    text_num[c] = cnt
                    num_text[str(cnt)] = c
                    cnt += 1
                logic = str(tmpp) + "(" + str(text_num[b]) + "," + str(text_num[c]) + ")"
            else:
                logic = str(tmpp) + "(" + str(text_num[b]) + ")"
            logic_sentence.append(logic)
        sentence.append(logic_sentence)
    return sentence


# 挖掘formula
def formula(left, right):
    tmp1 = extractor.triples_main(left)
    tmp2 = extractor.triples_main(right)
    tmp = []
    for each1 in tmp1:
        if not len(tmp1[each1]):
            continue
        for each in tmp1[each1]:
            if each[0] not in text_num.keys():
                text_num[each[0]] = len(text_num) + 1
                num_text[str(len(num_text) + 1)] = each[0]
                tmpp = len(text_num) + 1
            else:
                text_num[each[0] + '_' + str(len(text_num) + 1)] = len(text_num) + 1
                num_text[str(len(num_text) + 1)] = each[0] + '_' + str(len(num_text) + 1)
                tmpp = len(text_num) + 1
            if each[1] not in text_num.keys():
                text_num[each[1]] = len(text_num) + 1
                num_text[str(len(num_text) + 1)] = each[1]
            if len(each) == 3:
                if each[2] not in text_num.keys():
                    text_num[each[2]] = len(text_num) + 1
                    num_text[str(len(num_text) + 1)] = each[2]
                tmp.append(str(tmpp) + "(" + str(text_num[each[1]]) + "," + str(text_num[each[2]]) + ")")
            else:
                tmp.append(str(tmpp) + "(" + str(text_num[each[1]]) + ")")
    left1 = "^".join(tmp)
    tmp = []
    for each2 in tmp2:
        if not len(tmp2[each2]):
            continue
        for each in tmp2[each2]:
            if each[0] not in text_num.keys():
                text_num[each[0]] = len(text_num) + 1
                num_text[str(len(num_text) + 1)] = each[0]
                tmpp = len(text_num) + 1
            else:
                text_num[each[0] + '_' + str(len(text_num) + 1)] = len(text_num) + 1
                num_text[str(len(num_text) + 1)] = each[0] + '_' + str(len(num_text) + 1)
                tmpp = len(text_num) + 1
            if each[1] not in text_num.keys():
                text_num[each[1]] = len(text_num) + 1
                num_text[str(len(num_text) + 1)] = each[1]
            if len(each) == 3:
                if each[2] not in text_num.keys():
                    text_num[each[2]] = len(text_num) + 1
                    num_text[str(len(num_text) + 1)] = each[2]
                tmp.append(str(tmpp) + "(" + str(text_num[each[1]]) + "," + str(text_num[each[2]]) + ")")
            else:
                tmp.append(str(tmpp) + "(" + str(text_num[each[1]]) + ")")
    right1 = "^".join(tmp)
    if left1 != "" and right1 != "":
        return left1 + " => " + right1
    else:
        return ""


# 挖掘predicate
def predicate():
    for k in text_num:
        predicate_list[k] = "".join(list(postagger.postag([k])))


# 提取因果句挖掘formula
def extract_causality(file_name, article):
    i = 0
    extractor = causality_extract.CausalityExractor()
    with open(file_name, 'r', encoding='utf-8') as file_object:
        for each in file_object:
            if i == article:
                break
            datas = extractor.extract_main(each)
            for data in datas:
                left = ''.join([word.split('/')[0] for word in data['cause'].split(' ') if word.split('/')[0]])
                right = ''.join([word.split('/')[0] for word in data['effect'].split(' ') if word.split('/')[0]])
                tmp = formula(left, right)
                if tmp != "":
                    formula_list.add(tmp)
            i += 1


# 提取复合句挖掘formula
def extract_complex(file_name, article):
    i = 0
    extractor = complex_extract.EventsExtraction()
    with open(file_name, 'r', encoding='utf-8') as file_object:
        for each in file_object:
            if i == article:
                break
            datas = extractor.extract_main(each)
            for data in datas:
                left = data['tuples']['pre_part']
                right = data['tuples']['post_part']
                tmp = formula(left, right)
                if tmp != "":
                    formula_list.add(tmp)
            i += 1
    return formula_list


# 计算互信息挖掘formula
def mi_collect():
    extractor = collection_mi.MI_Train(5, sentence)
    tmp = extractor.mi_main()
    Index = list(chain.from_iterable(sentence))
    for each in tmp:
        for i in range(1, len(each)):
            if Index.index(each[0]) < Index.index(each[i]):
                formula_list.add(each[0] + " => " + each[i])
            else:
                formula_list.add(each[i] + " => " + each[0])


# 将数据写入文本
def write_txt():
    with open(markov_text_num, 'w', encoding='utf-8') as file2:
        for k, v in text_num.items():
            file2.write(k + ',' + str(v) + "\n")
    with open(markov_num_text, 'w', encoding='utf-8') as file3:
        for k, v in num_text.items():
            file3.write(str(k) + ',' + v + "\n")
    with open(markov_formula, 'w', encoding='utf-8') as file4:
        for each in formula_list:
            file4.write(each + "\n")
    with open(markov_predicate, 'w', encoding='utf-8') as file5:
        for k, v in predicate_list.items():
            file5.write(str(text_num[k]) + ',' + v + "\n")


# if __name__ == "__main__":
#     triple_text(file_name, 305, 1800)
#     dict_text()
#     mi_collect()
#     extract_causality(file_name, 305)
#     extract_complex(file_name, 305)
#     predicate()
#     write_txt()
