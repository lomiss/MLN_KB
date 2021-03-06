# -*- coding: utf-8 -*-
import os
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller


class LtpParser:
    def __init__(self):
        LTP_DIR = "ltp_data_v3.4.0/"
        self.segmentor = Segmentor()
        self.segmentor.load(os.path.join(LTP_DIR, "cws.model"))
        self.postagger = Postagger()
        self.postagger.load(os.path.join(LTP_DIR, "pos.model"))
        self.parser = Parser()
        self.parser.load(os.path.join(LTP_DIR, "parser.model"))
        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(os.path.join(LTP_DIR, "ner.model"))
        self.labeller = SementicRoleLabeller()
        self.labeller.load(os.path.join(LTP_DIR, 'pisrl_win.model'))

    '''语义角色标注'''
    def format_labelrole(self, words, postags, arcs):
        roles = self.labeller.label(words, postags, arcs)
        roles_dict = {}
        for role in roles:
            roles_dict[role.index] = {arg.name: [arg.name, arg.range.start, arg.range.end] for arg in role.arguments}
        return roles_dict

    '''句法分析---为句子中的每个词语维护一个保存句法依存儿子节点的字典'''
    def build_parse_child_dict(self, words, postags, arcs):
        child_dict_list = []
        format_parse_list = []
        for index in range(len(words)):
            child_dict = dict()
            for arc_index in range(len(arcs)):
                if arcs[arc_index].head == index + 1:  # arcs的索引从1开始
                    if arcs[arc_index].relation in child_dict:
                        child_dict[arcs[arc_index].relation].append(arc_index)
                    else:
                        child_dict[arcs[arc_index].relation] = []
                        child_dict[arcs[arc_index].relation].append(arc_index)
            child_dict_list.append(child_dict)
        rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
        relation = [arc.relation for arc in arcs]  # 提取依存关系
        heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # 匹配依存父节点词语
        for i in range(len(words)):
            # ['ATT', '李克强', 0, 'nh', '总理', 1, 'n']
            a = [relation[i], words[i], i, postags[i], heads[i], rely_id[i] - 1, postags[rely_id[i] - 1]]
            format_parse_list.append(a)
        return child_dict_list, format_parse_list

    '''parser主函数'''
    def parser_main(self, sentence):
        # 分词
        words = list(self.segmentor.segment(sentence))
        # 词性标注
        postags = list(self.postagger.postag(words))
        # 依存语义分析
        arcs = self.parser.parse(words, postags)
        # 命名实体识别
        netags = self.recognizer.recognize(words, postags)
        # 依存句法分析
        child_dict_list, format_parse_list = self.build_parse_child_dict(words, postags, arcs)
        # 语义角色标注
        roles_dict = self.format_labelrole(words, postags, arcs)
        return words, postags, child_dict_list, roles_dict, arcs, netags, format_parse_list


# if __name__ == '__main__':
#     parse = LtpParser()
#     sentence = '依存语法分析是一种通过依存关系揭示句法结构的方法'
#     words, postags, child_dict_list, roles_dict, arcs, netags, format_parse_list = parse.parser_main(sentence)
#     print("分词：===================================")
#     print(words)
#     print("词性标注：===================================")
#     for word, tag in zip(words, postags):
#         print(word + '/' + tag, end="|")
#     print("\n依存语义分析：===================================")
#     print("\t".join("%d:%s" % (arc.head-1, arc.relation) for arc in arcs))
#     print("命名实体识别：===================================")
#     for word, ntag in zip(words, netags):
#         print(word + '/' + ntag, end="|")
#     print("\n依存句法分析：===================================")
#     print(child_dict_list)
#     print(format_parse_list)
#     print("语义角色标注：===================================")
#     for each in roles_dict.items():
#         print(each)
