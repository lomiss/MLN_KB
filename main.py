# -*- coding: utf-8 -*-
import mln
import logci_extract
import sim_cilin
import sim_hownet
import pickle
import json
import numpy as np
import tensorflow as tf
from tkinter import *
from tkinter import filedialog
from datetime import datetime
from Tensor_QA.seq_to_seq import Seq2Seq
from Tensor_QA.data_utils import batch_flow

# 实例化object，建立窗口window
window = Tk()
window.resizable(width=False, height=False)
# 给窗口的可视化起名字
window.title('MLN知识推理问答工具V1.0')
# 设定窗口的大小(长 * 宽)
window.geometry('1000x600')  # 这里的乘是小x
extractor = logci_extract.segmentor
konwledge = mln.konwledge_modelling()
simer = sim_cilin.SimCilin(cilin_path="resource_cleaned/cilin.txt")
simer1 = sim_hownet.SimHownet(semantic_path="resource_cleaned/hownet.dat")
markov_formula = 'mln_data/markov_formula.txt'
markov_predicate = 'mln_data/markov_predicate.txt'
markov_text_num = 'mln_data/markov_text_num.txt'
markov_num_text = 'mln_data/markov_num_text.txt'
formula_list = []
predicate_dict = {}
text_num = {}
num_text = {}

logic = set()
predicate = set()
tmp_formula = []

state = {'1': 'Next：检查文本合法性', '2': 'Next：生成谓词', '3': 'Next：文本-数字转化', '4': 'Next：提取公式', '5': 'Next：写入数据', '6': '预处理完成'}
cur_state = '1'
filename = ""


# 读取本地文件
def init_():
    with open(markov_text_num, 'r', encoding='utf-8') as file2:
        for each in file2:
            text_num[each.strip("\n").split(",")[0]] = int(each.strip("\n").split(",")[1])
    with open(markov_num_text, 'r', encoding='utf-8') as file3:
        for each in file3:
            num_text[int(each.strip("\n").split(",")[0])] = each.strip("\n").split(",")[1]
    with open(markov_formula, 'r', encoding='utf-8') as file4:
        for each in file4:
            formula_list.append(each.strip("\n"))
    with open(markov_predicate, 'r', encoding='utf-8') as file5:
        for each in file5:
            predicate_dict[int(each.strip("\n").split(",")[0])] = each.strip("\n").split(",")[1]


# 求得声明谓词
def get_predicate(formu):
    atom = []
    tmp0 = formu.split('(')[0]
    tmp = formu.split('(')[1]
    tmp = tmp[:len(tmp) - 1]
    if ',' in tmp:
        a = int(tmp.split(',')[0])
        b = int(tmp.split(',')[1])
        tmp1 = tmp0 + '(' + predicate_dict[a] + ',' + predicate_dict[b] + ')'
        predicate.add(tmp1)
        atom.append(a)
        atom.append(b)
    else:
        tmp1 = tmp0 + '(' + predicate_dict[int(tmp)] + ')'
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


# 获取关键词相关谓词
def extract(content):
    # 无关谓词论元越多，推理的速度越慢
    # 规则越多，训练的速度越慢
    content = list(extractor.segment(content))
    after_content = []
    formula = []
    for each in content:
        same_text = []
        for text in text_num.keys():
            if str(text).split('_')[0] == each:
                same_text.append(str(text_num[text]))
        if len(same_text):
            after_content.append(same_text)
        else:
            return 0
    # 获取包含所有after_content的公式
    for each in formula_list:
        num = []
        tmp = ""
        for i in range(len(each)):
            if str(each[i]).isnumeric():
                tmp += each[i]
            else:
                if tmp != "":
                    num.append(tmp)
                    tmp = ""
        # 筛选formula
        flag = 1
        for same in after_content:
            flag1 = 0
            for i in same:
                if i in num:
                    flag1 = 1
                    break
            if flag1 == 0:
                flag = 0
                break
        if flag:
            formula.append(each)
    if not len(formula):
        return 0
    # 判断同一个公式中是否有相同论元
    for index in range(len(formula)):
        each = formula[index]
        l = 0
        atom = []
        others = []
        for i in range(len(each)):
            if each[i] == '(':
                others.append(each[l:i])
                others.append(each[i:i + 1])
                l = i + 1
            elif each[i] == ',':
                atom.append(num_text[int(each[l:i])])
                others.append(each[i:i + 1])
                l = i + 1
            elif each[i] == ')':
                atom.append(num_text[int(each[l:i])])
                others.append(each[i:i + 1])
                l = i + 1
        for j in range(len(atom)):
            for k in range(j + 1, len(atom)):
                if simer.compute_word_sim(atom[j], atom[k]) >= 0.9:
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
        formula[index] = new_formula
    # 对每个公式求得三元素
    for each in formula:
        atom_dict = {}
        all_atom = []
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
    return 1


# 将结果文本化
def formula_text(form):
    tmp0 = int(form.split('(')[0])
    tmp = form.split('(')[1]
    tmp = tmp[:len(tmp) - 1]
    if ',' in tmp:
        tmp1 = str(num_text[int(tmp.split(',')[0])]).split('_')[0] + \
               '|' + str(num_text[tmp0]).split('_')[0] + '|' + str(num_text[int(tmp.split(',')[1])]).split('_')[0]
    else:
        a = str(num_text[int(tmp)]).split('_')[0]
        b = str(num_text[tmp0]).split('_')[0]
        tmp1 = a + '|' + b + '(' + b + '|' + a + ')'
    return tmp1


# 展示推理结果
def show_infercnce(ans):
    for each in ans:
        tmp1 = []
        tmp2 = []
        for each1 in str(each).split(" => ")[0].split("^"):
            tmp1.append(formula_text(each1))
        for each1 in str(each).split(" => ")[1].split("^"):
            tmp2.append(formula_text(each1))
        t3.insert(END, "^".join(tmp1) + "=>" + "^".join(tmp2) + '\n')


# 开始知识抽取
def start_main():
    if not len(formula_list) or not len(predicate_dict):
        t3.insert(END, '本地未发现数据，请先通过预处理哦~\n')
        t3.see(END)
        return
    t3.insert(END, '================================================================\n')
    logic.clear()
    predicate.clear()
    tmp_formula.clear()
    start = datetime.now()
    content = v1.get()
    if content == "" or extract(content) == 0:
        t3.insert(END, '你的关键字未找到匹配项，可以尝试扩充语料库\n')
        t3.see(END)
        return
    rate = float(v5.get())
    if len(tmp_formula) > 5:
        t3.insert(END, '证据原子过多，有可能导致学习超时，进而内存爆掉哦~\n')
        t3.see(END)
        return
    predicates = konwledge.read_predicate(list(predicate))
    formula = konwledge.read_formula(tmp_formula)
    database = konwledge.read_data(list(logic))
    data, mln = konwledge.model_config(predicates, formula, database, 'markov.mln', 'markov.db')
    function_value = str(konwledge.activate_model(data, mln)[4][0])
    value = float(function_value.split(": ")[1])
    if value > float(v2.get()):
        t3.insert(END, '马尔科夫逻辑网的函数学习值大于了给定临界条件，提取过程中可能会运行很久，甚至内存爆掉，出于安全考虑停止运行,当然你可以尝试调高函数学习值\n')
        t3.see(END)
        return
    ans = konwledge.inference(tmp_formula, data, mln, rate)
    show_infercnce(ans)
    end = datetime.now()
    t3.insert(END, '【用时：' + str((end - start).seconds) + '】知识抽取成功.\n')
    t3.see(END)


# 更新状态
def update():
    Label(window, text=state[cur_state], fg='red', font=('Times New Roman', 12), width=18,
          height=1).place(x=353, y=377, anchor='nw')
    window.after(500, update)


# 打开文件
def open_file():
    global filename, cur_state
    t1.delete('1.0', END)
    filename = filedialog.askopenfilename(filetypes=[("TXT", ".txt"), ("python", ".py")])
    start = datetime.now()
    cnt = 0
    t1.insert(END, '打开文件成功，开始检验文本合法性...\n')
    try:
        with open(filename, encoding='utf-8') as file:
            for each in file:
                cnt += 1
                t1.insert(END, each)
                if each == '\n' or " " in each:
                    t1.insert(END, "【非法】单行为空白或者句子中出现空格\n")
                    cur_state = '1'
                    return
                else:
                    t1.insert(END, "OK...\n")
    except OSError as reason:
        t1.insert(END, '出错啦！' + str(reason) + "\n")
        cur_state = '1'
        return
    end = datetime.now()
    t1.insert(END, '【用时：' + str((end - start).seconds) + '秒】文本检验成功.共计' + str(cnt) + '行\n')
    cur_state = '2'
    t1.see(END)


# 生成谓词
def generate_logic():
    global cur_state
    second = int(v3.get())
    lines = int(v4.get())
    if cur_state == '2':
        start = datetime.now()
        triple = logci_extract.triple_text(filename, lines, second)
        if not len(triple):
            t2.insert(END, '运算时间超过临界时间，请重试或者增大临界时间\n')
        else:
            end = datetime.now()
            for each in triple:
                t2.insert(END, str(each) + '\n')
            t2.insert(END, '【用时：' + str((end - start).seconds) + '秒】生成谓词成功\n')
            cur_state = '3'
    elif cur_state > '2':
        t2.insert(END, '如果要重复前面操作，请从第一步开始\n')
    else:
        t2.insert(END, '请先通过文本合法性检测哦~\n')
    t2.see(END)


# 文本-数字转化
def discrete():
    global cur_state
    if cur_state == '3':
        start = datetime.now()
        sentence = logci_extract.dict_text()
        for each in sentence:
            t2.insert(END, str(each) + '\n')
        end = datetime.now()
        t2.insert(END, '【用时：' + str((end - start).seconds) + '秒】文本-数字转化成功\n')
        cur_state = '4'
    elif cur_state > '3':
        t2.insert(END, '如果要重复前面操作，请从第一步开始\n')
    else:
        t2.insert(END, '请先通过生成谓词哦~\n')
    t2.see(END)


# 提取公式
def extract_formula():
    global cur_state
    lines = int(v4.get())
    if cur_state == '4':
        start = datetime.now()
        logci_extract.mi_collect()
        logci_extract.extract_causality(filename, lines)
        formula_list_tmp = logci_extract.extract_complex(filename, lines)
        for each in formula_list_tmp:
            t2.insert(END, str(each) + '\n')
        end = datetime.now()
        t2.insert(END, '【用时：' + str((end - start).seconds) + '秒】提取公式成功\n')
        cur_state = '5'
    elif cur_state > '4':
        t2.insert(END, '如果要重复前面操作，请从第一步开始\n')
    else:
        t2.insert(END, '请先文本-数字转化哦~\n')
    t2.see(END)


# 写入数据
def wirte_data():
    global cur_state
    if cur_state == '5':
        start = datetime.now()
        logci_extract.predicate()
        logci_extract.write_txt()
        end = datetime.now()
        t2.insert(END, '【用时：' + str((end - start).seconds) + '秒】写入数据成功\n')
        cur_state = '6'
    elif cur_state > '5':
        t2.insert(END, '如果要重复前面操作，请从第一步开始\n')
    else:
        t2.insert(END, '请先提取公式哦~\n')
    t2.see(END)
    init()


def start_qa():
    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)
        user_text = v6.get()
        x_test = [list(user_text.lower())]
        bar = batch_flow([x_test], ws, 1)
        x, xl = next(bar)
        x = np.flip(x, axis=1)
        pred = model_pred.predict(sess, np.array(x), np.array(xl))
        for p in pred:
            ans = ws.inverse_transform(p)
            v7.set("".join(ans).strip("</S>"))


if __name__ == "__main__":
    # ===============Tensor预处理阶段=================
    params = json.load(open('QA_data/params.json'))
    with open('QA_data/chatbot.pkl', 'rb') as input_file:
        try:
            pickle.load(input_file)
        except EOFError:
            print("chatbot.pkl is empty")

    ws = pickle.load(open('QA_data/ws.pkl', 'rb'))

    config = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = 'model/s2ss_chatbot.ckpt'

    tf.reset_default_graph()
    model_pred = Seq2Seq(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=0,
        **params
    )
    init = tf.global_variables_initializer()
    # ==============================================
    init_()
    update()
    # 【声明阶段】
    l0 = Label(window,
               text='【注意】本工具为科研工具，尚在试验阶段，存在Bug，运算速度方面\n非常依赖计算机的配置，时间随着数据量的增大而变长，运算过程中\n程序会出现未响应的情况，为正常现象，由于知识抽取算法运用了马\n尔科夫逻辑网，数据量一大，其运行时间指数级增长，无关论元过多\n会出现爆栈的情况，因此需要设定临界时间以强制结束且算法只适合\n简短句子的抽取，不过作者会继续改进算法，优化性能'
               , fg='black', font=('Times New Roman', 12), width=55, height=6).place(x=10, y=15, anchor='nw')
    # 【打开文件阶段】
    b1 = Button(window, text='打开文件', font=('Times New Roman', 12),
                width=10, height=1, command=open_file).place(x=10, y=147, anchor='nw')
    l1 = Label(window, text='Have a nice day', fg='black', font=('Times New Roman', 15), width=15,
               height=1).place(x=200, y=152, anchor='nw')
    t1 = Text(window)
    t1.place(x=10, y=187, anchor='nw', relwidth=0.5, relheight=0.3)
    sb1 = Scrollbar(t1, command=t1.yview)
    sb1.pack(side=RIGHT, fill=Y)
    t1.config(yscrollcommand=sb1.set)
    # 【以下是预处理阶段】
    v4 = StringVar()
    v4.set(20)
    l2 = Label(window, text='处理行数：', fg='black', font=('Times New Roman', 12), width=10,
               height=1).place(x=8, y=377, anchor='nw')
    e1 = Entry(window, textvariable=v4, width=10).place(x=98, y=379, anchor='nw')
    l5 = Label(window, text='临界时间(单位秒)：', fg='black', font=('Times New Roman', 12), width=18,
               height=1).place(x=158, y=377, anchor='nw')
    v3 = StringVar()
    v3.set(1800)
    e4 = Entry(window, textvariable=v3, width=5).place(x=310, y=380, anchor='nw')
    l6 = Label(window, text=state[cur_state], fg='red', font=('Times New Roman', 12), width=18,
               height=1).place(x=353, y=377, anchor='nw')
    b2 = Button(window, text='生成谓词', font=('Times New Roman', 12),
                width=12, height=1, command=generate_logic).place(x=405, y=420, anchor='nw')
    b3 = Button(window, text='文本-数字转化', font=('Times New Roman', 12),
                width=12, height=1, command=discrete).place(x=405, y=460, anchor='nw')
    b4 = Button(window, text='提取公式', font=('Times New Roman', 12),
                width=12, height=1, command=extract_formula).place(x=405, y=500, anchor='nw')
    b5 = Button(window, text='写入数据', font=('Times New Roman', 12),
                width=12, height=1, command=wirte_data).place(x=405, y=540, anchor='nw')
    t2 = Text(window)
    t2.place(x=10, y=410, anchor='nw', relwidth=0.39, relheight=0.3)
    sb2 = Scrollbar(t2, command=t2.yview)
    sb2.pack(side=RIGHT, fill=Y)
    t2.config(yscrollcommand=sb2.set)
    # 【以下是知识提取阶段】
    l3 = Label(window, text='提取的关键词：', fg='black', font=('Times New Roman', 12), width=12,
               height=1).place(x=530, y=17, anchor='nw')
    v1 = StringVar()
    e2 = Entry(window, textvariable=v1, width=42).place(x=640, y=20, anchor='nw')
    l4 = Label(window, text='函数学习值：', fg='black', font=('Times New Roman', 12), width=12,
               height=1).place(x=520, y=55, anchor='nw')
    v2 = StringVar()
    v2.set(6.0)
    e3 = Entry(window, textvariable=v2, width=6).place(x=620, y=57, anchor='nw')
    l7 = Label(window, text='提取权重>=：', fg='black', font=('Times New Roman', 12), width=12,
               height=1).place(x=720, y=55, anchor='nw')
    v5 = StringVar()
    v5.set(0.5)
    e5 = Entry(window, textvariable=v5, width=6).place(x=825, y=57, anchor='nw')
    b6 = Button(window, text='开始提取', font=('Times New Roman', 12),
                width=10, height=1, command=start_main).place(x=890, y=50, anchor='nw')
    t3 = Text(window)
    t3.place(x=530, y=94, anchor='nw', relwidth=0.46, relheight=0.66)
    sb3 = Scrollbar(t3, command=t3.yview)
    sb3.pack(side=RIGHT, fill=Y)
    t3.config(yscrollcommand=sb3.set)
    v6 = StringVar()
    l8 = Label(window, text='你的问题:', fg='black', font=('Times New Roman', 12), width=7,
               height=1).place(x=550, y=510, anchor='nw')
    e6 = Entry(window, textvariable=v6, width=42).place(x=620, y=510, anchor='nw')
    v7 = StringVar()
    l9 = Label(window, text='Robot:', fg='black', font=('Times New Roman', 12), width=7,
               height=1).place(x=550, y=550, anchor='nw')
    e7 = Entry(window, textvariable=v7, width=42).place(x=620, y=550, anchor='nw')
    b7 = Button(window, text='Search', font=('Times New Roman', 12),
                width=6, height=1, command=start_qa).place(x=930, y=525, anchor='nw')
    # 【主窗口循环显示】
    window.mainloop()
