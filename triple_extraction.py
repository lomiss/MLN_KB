# -*- coding: utf-8 -*-
from sentence_parser import *
import re


class TripleExtractor:
    def __init__(self):
        self.parser = LtpParser()

    # 文章分句处理, 切分长句，冒号，分号，感叹号等做切分标识
    def split_sents(self, content):
        return [sentence for sentence in re.split(r'[？?！!。；;：:\n\r]', content) if sentence]

    # 利用语义角色标注,直接获取主谓宾三元组,基于A0,A1,A2
    def ruler1(self, words, postags, roles_dict, role_index):
        v = words[role_index]
        role_info = roles_dict[role_index]
        if 'A0' in role_info.keys() and 'A1' in role_info.keys():
            s = ''.join([words[word_index] for word_index in range(role_info['A0'][1], role_info['A0'][2] + 1) if
                         postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]])
            o = ''.join([words[word_index] for word_index in range(role_info['A1'][1], role_info['A1'][2] + 1) if
                         postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]])
            if s and o:
                return '1', [v, s, o]
        elif 'A0' in role_info.keys():
            s = ''.join([words[word_index] for word_index in range(role_info['A0'][1], role_info['A0'][2] + 1) if
                         postags[word_index][0] not in ['w', 'u', 'x']])
            if s:
                return '2', [v, s]
        elif 'A1' in role_info.keys():
            o = ''.join([words[word_index] for word_index in range(role_info['A1'][1], role_info['A1'][2]+1) if
                         postags[word_index][0] not in ['w', 'u', 'x']])
            return '3', [v, o]
        return '4', []

    # 三元组抽取主函数
    def ruler2(self, words, postags, child_dict_list, arcs, roles_dict):
        svos = []
        for index in range(len(postags)):
            tmp = 1
            # 先借助语义角色标注的结果，进行多元组抽取
            if index in roles_dict:
                flag, triple = self.ruler1(words, postags, roles_dict, index)
                if flag != '4':
                    svos.append(triple)
                    tmp = 0
            if tmp == 1:
                # 如果语义 角色标记为空，则使用依存句法进行抽取
                # if postags[index] == 'v':
                if postags[index]:
                    # 抽取以谓词为中心的事实多元组
                    child_dict = child_dict_list[index]
                    # 主谓宾
                    if 'SBV' in child_dict and 'VOB' in child_dict:
                        r = words[index]
                        e1 = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                        e2 = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                        svos.append([r, e1, e2])
                    # 定语后置，动宾关系
                    relation = arcs[index].relation
                    head = arcs[index].head
                    if relation == 'ATT':
                        if 'VOB' in child_dict:
                            e1 = self.complete_e(words, postags, child_dict_list, head - 1)
                            r = words[index]
                            e2 = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                            temp_string = r + e2
                            if temp_string == e1[:len(temp_string)]:
                                e1 = e1[len(temp_string):]
                            if temp_string not in e1:
                                svos.append([r, e1, e2])
                    # 含有介宾关系的主谓动补关系
                    if 'SBV' in child_dict and 'CMP' in child_dict:
                        e1 = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                        cmp_index = child_dict['CMP'][0]
                        r = words[index] + words[cmp_index]
                        if 'POB' in child_dict_list[cmp_index]:
                            e2 = self.complete_e(words, postags, child_dict_list, child_dict_list[cmp_index]['POB'][0])
                            # eg：['综合因素', '构成在', '一起']['国内玉米市场', '发展向', '何处']['玉米', '投放到', '市场']
                            svos.append([r, e1, e2])
        return svos

    # 对找出的主语或者宾语进行扩展
    def complete_e(self, words, postags, child_dict_list, word_index):
        child_dict = child_dict_list[word_index]
        prefix = ''
        if 'ATT' in child_dict:
            for i in range(len(child_dict['ATT'])):
                prefix += self.complete_e(words, postags, child_dict_list, child_dict['ATT'][i])
        postfix = ''
        if postags[word_index] == 'v':
            if 'VOB' in child_dict:
                postfix += self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
            if 'SBV' in child_dict:
                prefix = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0]) + prefix

        return prefix + words[word_index] + postfix

    # 程序主控函数
    def triples_main(self, content):
        sentences = self.split_sents(content)
        sentence_svos = {}
        for sentence in sentences:
            svos = []
            words, postags, child_dict_list, roles_dict, arcs, netags, format_parse_list = self.parser.parser_main(
                sentence)
            svo = self.ruler2(words, postags, child_dict_list, arcs, roles_dict)
            svos += svo
            sentence_svos[sentence] = svos
        return sentence_svos

# if __name__ == "__main__":
#     content1 = "快死的叶子发黄而卷曲"
#     extractor = TripleExtractor()
#     sentence_svos = extractor.triples_main(content1)
#     for each in sentence_svos:
#         print(each, sentence_svos[each])