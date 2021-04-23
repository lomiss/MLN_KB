# -*- coding: utf-8 -*-
import os
import mln
import json
import sim_cilin
import sim_hownet
from pyltp import Postagger
import triple_extraction
import causality_extract
import complex_extract
konwledge = mln.konwledge_modelling()
LTP_DIR = "ltp_data_v3.4.0"
postagger = Postagger()
postagger.load(os.path.join(LTP_DIR, "pos.model"))
extractor = triple_extraction.TripleExtractor()
extractor1 = causality_extract.CausalityExractor()
extractor2 = complex_extract.EventsExtraction()
simer = sim_cilin.SimCilin(cilin_path="resource_cleaned/cilin.txt")
simer1 = sim_hownet.SimHownet(semantic_path='resource_cleaned/hownet.dat')
entailment = []
neutral = []
contradiction = []
ans_success = []
ans_loss = []
ans_unmatch = []
ans_timeout = []
cnt_success = 0
cnt_loss = 0
cnt_timeout = 0
cnt_unmatch = 0
degree0 = 0
degree20 = 0
degree40 = 0
degree60 = 0
degree80 = 0
sim_below_80 = 0
sim_above_80 = 0


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
            else:
                text_num[each[0] + '_' + str(len(text_num) + 1)] = len(text_num) + 1
                num_text[str(len(num_text) + 1)] = each[0] + '_' + str(len(num_text) + 1)
            if each[1] not in text_num.keys():
                text_num[each[1]] = len(text_num) + 1
                num_text[str(len(num_text) + 1)] = each[1]
            if len(each) == 3:
                if each[2] not in text_num.keys():
                    text_num[each[2]] = len(text_num) + 1
                    num_text[str(len(num_text) + 1)] = each[2]
                tmp.append(str(len(text_num)) + "(" + str(text_num[each[1]]) + "," + str(text_num[each[2]]) + ")")
            else:
                tmp.append(str(len(text_num)) + "(" + str(text_num[each[1]]) + ")")
    left1 = "^".join(tmp)
    tmp = []
    for each2 in tmp2:
        if not len(tmp2[each2]):
            continue
        for each in tmp2[each2]:
            if each[0] not in text_num.keys():
                text_num[each[0]] = len(text_num) + 1
                num_text[str(len(num_text) + 1)] = each[0]
            else:
                text_num[each[0] + '_' + str(len(text_num) + 1)] = len(text_num) + 1
                num_text[str(len(num_text) + 1)] = each[0] + '_' + str(len(num_text) + 1)
            if each[1] not in text_num.keys():
                text_num[each[1]] = len(text_num) + 1
                num_text[str(len(num_text) + 1)] = each[1]
            if len(each) == 3:
                if each[2] not in text_num.keys():
                    text_num[each[2]] = len(text_num) + 1
                    num_text[str(len(num_text) + 1)] = each[2]
                tmp.append(str(len(text_num)) + "(" + str(text_num[each[1]]) + "," + str(text_num[each[2]]) + ")")
            else:
                tmp.append(str(len(text_num)) + "(" + str(text_num[each[1]]) + ")")
    right1 = "^".join(tmp)
    if left1 != "" and right1 != "":
        return left1 + " => " + right1
    else:
        return ""


# 提取因果句挖掘formula
def extract_causality(text):
    datas = extractor1.extract_main(text)
    for data in datas:
        left = ''.join([word.split('/')[0] for word in data['cause'].split(' ') if word.split('/')[0]])
        right = ''.join([word.split('/')[0] for word in data['effect'].split(' ') if word.split('/')[0]])
        tmp = formula(left, right)
        if tmp != "":
            formula_list.append(tmp)


# 提取复合句挖掘formula
def extract_complex(text):
    datas = extractor2.extract_main(text)
    for data in datas:
        left = data['tuples']['pre_part']
        right = data['tuples']['post_part']
        tmp = formula(left, right)
        if tmp != "":
            formula_list.append(tmp)


# 挖掘predicate
def extract_predicate():
    for k in text_num:
        pos = "".join(list(postagger.postag([k])))
        if pos == "":
            return 0
        predicate_list[int(text_num[k])] = pos


# 求得声明谓词
def get_predicate(formu):
    atom = []
    tmp0 = formu.split('(')[0]
    tmp = formu.split('(')[1]
    tmp = tmp[:len(tmp) - 1]
    if ',' in tmp:
        a = int(tmp.split(',')[0])
        b = int(tmp.split(',')[1])
        tmp1 = tmp0 + '(' + predicate_list[a] + ',' + predicate_list[b] + ')'
        predicate.add(tmp1)
        atom.append(a)
        atom.append(b)
    else:
        tmp1 = tmp0 + '(' + predicate_list[int(tmp)] + ')'
        predicate.add(tmp1)
        atom.append(int(tmp))
    return atom


# 求得规则
def get_formula(formu, dict):
    tmp0 = formu.split('(')[0]
    tmp = formu.split('(')[1]
    tmp = tmp[:len(tmp) - 1]
    if ',' in tmp:
        a = int(tmp.split(',')[0])
        b = int(tmp.split(',')[1])
        tmp1 = tmp0 + '(' + dict[a] + ',' + dict[b] + ')'
    else:
        tmp1 = tmp0 + '(' + dict[int(tmp)] + ')'
    return tmp1


def start():
    for index in range(len(formula_list)):
        each = formula_list[index]
        l = 0
        atom = []
        others = []
        for i in range(len(each)):
            if each[i] == '(':
                others.append(each[l:i])
                others.append(each[i:i+1])
                l = i + 1
            elif each[i] == ',':
                atom.append(num_text[each[l:i]])
                others.append(each[i:i+1])
                l = i + 1
            elif each[i] == ')':
                atom.append(num_text[each[l:i]])
                others.append(each[i:i+1])
                l = i + 1
        for j in range(len(atom)):
            for k in range(j+1, len(atom)):
                if simer1.distance(atom[j], atom[k]) >= 0.9:
                    atom[k] = atom[j]
        new_formula = ""
        cnt = 0
        for other in others:
            if other == '(' or other == ',':
                new_formula += other
                new_formula += str(text_num[atom[cnt]])
                cnt += 1
            else:
                new_formula += other
        formula_list[index] = new_formula
    # 对每个公式求得三元素
    for each in formula_list:
        atom_dict = {}
        all_atom = []
        if " => " in each:
            left = str(each).split(" => ")[0].split("^")
            right = str(each).split(" => ")[1].split("^")
            # 获取谓词声明
            for each1 in left:
                logic.add(each1)
                all_atom += get_predicate(each1)
            for each1 in right:
                logic.add(each1)
                all_atom += get_predicate(each1)
            cnt = 1
            for each2 in all_atom:
                if each2 not in atom_dict:
                    atom_dict[each2] = 'p' + str(cnt)
                    cnt += 1
            # 获取公式
            tmp1 = []
            tmp2 = []
            for each3 in left:
                tmp1.append(get_formula(each3, atom_dict))
            for each3 in right:
                tmp2.append(get_formula(each3, atom_dict))
            tmp_formula.append("^".join(tmp1) + " => " + "^".join(tmp2))
        else:
            logic.add(each)
            tmp1 = each.split('(')[0]
            tmp2 = each.split('(')[1]
            tmp2 = tmp2[:len(tmp2) - 1]
            if ',' in tmp2:
                a = int(tmp2.split(',')[0])
                b = int(tmp2.split(',')[1])
                tmpp = tmp1 + '(p1,p2)'
                tmppp = tmp1 + '(' + predicate_list[a] + ',' + predicate_list[b] + ')'
            else:
                tmpp = tmp1 + '(p1)'
                tmppp = tmp1 + '(' + predicate_list[int(tmp2)] + ')'
            predicate.add(tmppp)
            tmp_formula.append(tmpp)


# 将结果文本化
def formula_text(form):
    tmp0 = form.split('(')[0]
    tmp00 = form.split('(')[1]
    tmp00 = tmp00[:len(tmp00)-1]
    if ',' in tmp00:
        tmp1 = str(num_text[tmp00.split(',')[0]]).split('_')[0] + str(num_text[tmp0]).split('_')[0] + str(num_text[tmp00.split(',')[1]]).split('_')[0]
    else:
        a = str(num_text[tmp00]).split('_')[0]
        b = str(num_text[tmp0]).split('_')[0]
        tmp1 = a + b
    return tmp1


# 展示推理结果
def show_infercnce(ans):
    tmp1 = []
    tmp2 = []
    left = str(ans).split(" => ")[0]
    if "^" in left:
        left = left[1:len(left)-1]
    left = left.split(" ^ ")
    for each1 in left:
        tmp1.append(formula_text(each1))
    if "=>" in ans:
        right = str(ans).split(" => ")[1]
        if "^" in right:
            right = right[1:len(right) - 1]
        right = right.split(" ^ ")
        for each1 in right:
            tmp2.append(formula_text(each1))
    return "".join(tmp1) + "。" + "".join(tmp2)


if __name__ == "__main__":
    # 读取文本
    # with open('resource_cleaned/cnsd-mnli/cnsd_multil_dev_matched.jsonl', 'r', encoding="utf-8") as file:
    #     for each in file:
    #         dictinfo = json.loads(each)
    #         if dictinfo["gold_label"] == "entailment":
    #             entailment.append([dictinfo["sentence1"], dictinfo["sentence2"]])
    #         if dictinfo["gold_label"] == "neutral":
    #             neutral.append([dictinfo["sentence1"], dictinfo["sentence2"]])
    #         if dictinfo["gold_label"] == "contradiction":
    #             contradiction.append([dictinfo["sentence1"], dictinfo["sentence2"]])
    with open('resource_cleaned/STS-B/cnsd-sts-test.txt', 'r', encoding="utf-8") as file:
        for each in file:
            tmp = each.split("||")
            if tmp[3] == '5\n' or tmp[3] == '4\n':
                entailment.append([tmp[1], tmp[2]])
            if tmp[3] == '3\n':
                neutral.append([tmp[1], tmp[2]])
            if tmp[3] == '1\n' or tmp[3] == '2\n':
                contradiction.append([tmp[1], tmp[2]])
    # 遍历文本对
    for i in range(len(contradiction)):
        print("================================================")
        # 初始变量定义
        cur_sentence = contradiction[i]
        predicate_list = {}
        text_num = {}
        num_text = {}
        formula_list = []
        formula_infer = []
        logic = set()
        predicate = set()
        tmp_formula = []
        cnt = 1
        each1 = []
        each2 = []
        print(i, cur_sentence)
        each0 = extractor.triples_main(cur_sentence[0])
        for j in each0:
            each1.extend(each0[j])
        each0 = extractor.triples_main(cur_sentence[1])
        for j in each0:
            each2.extend(each0[j])
        # 获取两个句子的谓词
        if len(each1) == 0:
            # 谓词丢失
            print("predicate_loss")
            cnt_loss += 1
            continue
        each1 = str(each1)
        tmp = each1[2:len(each1) - 2].split('], [')
        logic_sentence = []
        tmp_logic = ""
        for each_triple in tmp:
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
                tmp_logic = str(tmpp) + "(" + str(text_num[b]) + "," + str(text_num[c]) + ")"
            else:
                tmp_logic = str(tmpp) + "(" + str(text_num[b]) + ")"
            formula_list.append(tmp_logic)
        extract_causality(cur_sentence[0])
        extract_complex(cur_sentence[0])
        if extract_predicate() == 0:
            print("predicate_loss")
            cnt_loss += 1
            continue
        start()
        if len(text_num) >= 15:
            cnt_timeout += 1
            print("timeout")
            continue
        predicates = konwledge.read_predicate(list(predicate))
        formulas = konwledge.read_formula(tmp_formula)
        database = konwledge.read_data(list(logic))
        data, mln = konwledge.model_config(predicates, formulas, database, 'markov.mln', 'markov.db')
        function_value = str(konwledge.activate_model(data, mln)[4][0])
        value = float(function_value.split(": ")[1])
        print(value)
        if value >= 6.0:
            cnt_timeout += 1
            print("timeout")
            continue
        # 推理句验证
        ans = konwledge.inference(tmp_formula, data, mln, 0)
        flag = 0
        mav_sim = 0
        for k, v in ans.items():
            if v >= 0.8:
                word = show_infercnce(k)
                sim = simer1.distance(word, cur_sentence[1])
                mav_sim = max(mav_sim, sim)
                for each in each2:
                    last_word = each[1] + each[0]
                    if len(each) == 3:
                        last_word += each[2]
                    sim = simer1.distance(word, last_word)
                    mav_sim = max(mav_sim, sim)
        if mav_sim <= 0.2:
            degree0 += 1
        elif mav_sim <= 0.4:
            degree20 += 1
        elif mav_sim <= 0.6:
            degree40 += 1
        elif mav_sim <= 0.8:
            degree60 += 1
        elif mav_sim <= 1.0:
            degree80 += 1
        # 原句验证
        sim = simer1.distance(cur_sentence[0], cur_sentence[1])
        if sim < 0.8:
            sim_below_80 += 1
        else:
            sim_above_80 += 1
        if mav_sim >= 0.8 or sim >= 0.8:
            flag = 1
            cnt_success += 1
            print("Match")
        if not flag:
            print("MisMatch")
            cnt_unmatch += 1
        ans_success.append(cnt_success)
        ans_loss.append(cnt_loss)
        ans_unmatch.append(cnt_unmatch)
        ans_timeout.append(cnt_timeout)
print(ans_success)
print(ans_unmatch)
print(ans_loss)
print(ans_timeout)
print("degree0：", degree0)
print("degree20：", degree20)
print("degree40：", degree40)
print("degree60：", degree60)
print("degree80：", degree80)
print("sim_above_80：", sim_above_80)
print("sim_below_80：", sim_below_80)
print("绝对推理成功率：", (cnt_success-sim_above_80)/sim_below_80)
print("相对推理成功率：", cnt_success/(cnt_success+cnt_unmatch))
