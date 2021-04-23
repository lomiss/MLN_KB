# -*- coding: utf-8 -*-
import collections
import math
import jieba.posseg as pseg


class MI_Train:
    def __init__(self, window_size, sentences):
        self.window_size = window_size
        self.sentences = sentences

    # 对语料进行处理
    def build_corpus_(self):
        def cut_words(sent):
            return [word.word for word in pseg.cut(sent) if word.flag[0] not in ['x', 'w', 'p', 'u', 'c']]
        # sentences = sentences[sent.split(' ') for sent in open(self.filepath).read().split('\n')]，若处理英文语料则使用这种方法
        sentences = [cut_words(sent) for sent in open(self.filepath, encoding="UTF-8").read().split('\n')][:2]
        return sentences

    # 统计词频
    def count_words(self, sentences):
        words_all = list()
        for sent in sentences:
            words_all.extend(sent)
        word_dict = {item[0]: item[1] for item in collections.Counter(words_all).most_common()}
        return word_dict, len(words_all)

    # 读取训练语料
    def build_cowords(self, sentences):
        train_data = list()
        for sent in sentences:
            for index, word in enumerate(sent):
                if index < self.window_size:
                    left = sent[:index]
                else:
                    left = sent[index - self.window_size: index]
                if index + self.window_size > len(sent):
                    right = sent[index + 1:]
                else:
                    right = sent[index + 1: index + self.window_size + 1]
                data = left + right + [sent[index]]
                if '' in data:
                    data.remove('')
                train_data.append(data)
        return train_data

    # 统计共现次数
    def count_cowords(self, train_data):
        co_dict = dict()
        # print(len(train_data))
        for index, data in enumerate(train_data):
            for index_pre in range(len(data)):
                for index_post in range(len(data)):
                    if data[index_pre] not in co_dict:
                        co_dict[data[index_pre]] = data[index_post]
                    else:
                        co_dict[data[index_pre]] += '@' + data[index_post]
        return co_dict

    # 计算互信息
    def compute_mi(self, word_dict, co_dict, sum_tf):
        def compute_mi(p1, p2, p12):
            return math.log2(p12) - math.log2(p1) - math.log2(p2)

        def build_dict(words):
            return {item[0]: item[1] for item in collections.Counter(words).most_common()}
        mis_dict = dict()
        for word, co_words in co_dict.items():
            co_word_dict = build_dict(co_words.split('@'))
            mi_dict = {}
            for co_word, co_tf in co_word_dict.items():
                if co_word == word:
                    continue
                p1 = word_dict[word] / sum_tf
                p2 = word_dict[co_word] / sum_tf
                p12 = co_tf / sum_tf
                mi = compute_mi(p1, p2, p12)
                mi_dict[co_word] = mi
            mi_dict = sorted(mi_dict.items(), key=lambda asd: asd[1], reverse=True)
            mis_dict[word] = mi_dict
        return mis_dict

    # 保存互信息文件
    def save_mi(self, mis_dict):
        f = open(self.mipath, 'w+')
        for word, co_words in mis_dict.items():
            co_infos = [item[0] + '@' + str(item[1]) for item in co_words]
            f.write(word + '\t' + ','.join(co_infos) + '\n')
        f.close()

    # 运行主函数
    def mi_main(self):
        # print('step 1/6: build corpus ..........')
        # sentences = self.build_corpus()
        # print('step 1/5: compute worddict..........')
        word_dict, sum_tf = self.count_words(self.sentences)
        # print('step 2/5: build cowords..........')
        train_data = self.build_cowords(self.sentences)
        # print('step 3/5: compute coinfos..........')
        co_dict = self.count_cowords(train_data)
        # print('step 4/5: compute words mi..........')
        mi_data = self.compute_mi(word_dict, co_dict, sum_tf)
        # print('step 5/5: save words mi..........')
        all_tmp = []
        for word, co_words in mi_data.items():
            tmp = []
            if len(co_words) >= 3:
                tmp.append(word)
                for item in co_words[:2]:
                    tmp.append(item[0])
                all_tmp.append(tmp)
        return all_tmp


#
# if __name__ == '__main__':
#     sentences = [['1(2,3)'], ['4(5,6)', '7(8,9)', '10(11,12)'], ['13(14,15)', '16(17)', '18(19)'],
#                  ['1(20,21)', '22(23)', '24(25,26)', '27(28,29)', '30(31)', '32(25)', '33(25)'],
#                  ['34(35)', '12(36)', '10(37,12)'],
#                  ['38(39)', '32(37)']]
#     window_size = 5
#     mier = MI_Train(window_size, sentences)
#     mier.mi_main()
