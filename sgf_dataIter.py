# coding: utf-8
import os
import sys
import time
# 可视化棋谱
from gobang_board_utils import chessboard, evaluation, searcher, psyco_speedup
# 加速函数 
psyco_speedup()

LETTER_NUM = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']
BIG_LETTER_NUM = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
NUM_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# 棋盘字母位置速查表
seq_lookup = dict(zip(LETTER_NUM, NUM_LIST))
num2char_lookup = dict(zip(NUM_LIST, BIG_LETTER_NUM))

# SGF文件
sgf_home = '/Users/anxingle/Downloads/SGF_Gomoku/sgf/'


def get_files_as_list(data_dir):
    # 扫描某目录下SGF文件列表
    file_list = os.listdir(data_dir)
    file_list = [item for item in file_list if item.endswith('.sgf') and os.path.isfile(os.path.join(data_dir, item))]
    return file_list

def content_to_order(sequence):
    # 棋谱字母转整型数字

    global seq_lookup   # 棋盘字母位置速查表
    seq_list = sequence.split(';')
    # list:['hh', 'ii', 'hi'....]
    seq_list = [item[2:4] for item in seq_list]
    # list: [112, 128, ...]
    seq_num_list = [seq_lookup[item[0]]*15+seq_lookup[item[1]] for item in seq_list]
    return seq_list, seq_num_list


def num2char(order_):
    global num2char_lookup
    Y_axis = num2char_lookup[order_/15]
    X_axis = num2char_lookup[order_ % 15]
    return '%s%s' % (Y_axis, X_axis)

def get_data_from_files(file_name, data_dir):
    """ 根据文件名读取SGF棋谱内容 """
    assert file_name.endswith('.sgf'), 'file: %s 不是SGF文件' % file_name
    with open(os.path.join(data_dir, file_name)) as f:
        p = f.read()
        # 棋谱内容 开始/结束 位置
        start = file_name.index('_') + 1
        end = file_name.index('_.')

        sequence = p[p.index('SZ[15]')+7:-4]
        try:
            seq_list, seq_num_list = content_to_order(sequence)
        except Exception as e:
            print('***' * 20)
            print(e)
            print(file_name)
        if file_name[file_name.index('_')+1:file_name.index('_')+6] == 'Blank' or file_name[file_name.index('_')+1:file_name.index('_')+6] == 'blank':
            winner = 1 
        if file_name[file_name.index('_')+1:file_name.index('_')+6] == 'White' or file_name[file_name.index('_')+1:file_name.index('_')+6] == 'white':
            winner = 2
        return {'winner': winner, 'seq_list': seq_list, 'seq_num_list': seq_num_list, 'file_name':file_name}


def read_files(data_dir):
    # 迭代读取目录下SGF文件的棋谱内容

    # 扫描获取data_dir目录下所有SGF文件
    file_list = get_files_as_list(data_dir)
    index = 0
    while True:
        if index >= len(file_list): yield None
        with open(data_dir+file_list[index]) as f:
            p = f.read()
            # 棋谱内容 开始/结束 位置
            start = file_list[index].index('_') + 1
            end = file_list[index].index('_.')

            sequence = p[p.index('SZ[15]')+7:-4]
            try:
                seq_list, seq_num_list = content_to_order(sequence)
            except Exception as e:
                print('***' * 20)
                print(e)
                print(file_list[index])
        if sequence[-5] == 'B' or sequence[-5] == 'b':
            winner = 1 
        if sequence[-5] == 'W' or sequence[-5] == 'w':
            winner = 2
        yield {'winner': winner, 'seq_list': seq_list, 'seq_num_list': seq_num_list, 'index': index, 'file_name':file_list[index]}
        index += 1


def gamemain(seq_list):
    b = chessboard()
    s = searcher()
    s.board = b.board()

    opening = [
        '1:HH 2:II',
        # '2:IG 2:GI 1:HH',
        # '1:IH 2:GI',
        # '1:HG 2:HI',
        # '2:HG 2:HI 1:HH',
        # '1:HH 2:IH 2:GI',
        # '1:HH 2:IH 2:HI',
        # '1:HH 2:IH 2:HJ',
        # '1:HG 2:HH 2:HI',
        # '1:GH 2:HH 2:HI',
    ]

    import random
    # 开局棋盘局面
    # openid = random.randint(0, len(opening) - 1)
    # 开局局面为黑方的第一手
    def num2char(order_):
        global num2char_lookup
        Y_axis = num2char_lookup[order_/15]
        X_axis = num2char_lookup[order_ % 15]
        return '%s%s' % (Y_axis, X_axis)


    start_open = '1:%s 2:%s' %(num2char(seq_list[0]), num2char(seq_list[1]))
    b.loads(start_open)
    turn = 2
    history = []
    undo = False

    # 设置难度
    DEPTH = 1
    # 对弈开始，从第黑方第二开始
    index = 2

    while True:
        print ''
        while 1:
            print '<ROUND %d>' % (len(history) + 1)
            b.show()
            print '该你移动了： (u:悔棋, q:退出):',
            # 默认一直继续下去
            text = raw_input().strip('\r\n\t ')
            text = '%s' % num2char(seq_list[index])
            print 'char:  ', num2char(seq_list[index])
            print 'num:  ', seq_list[index]
            index += 1
            if len(text) == 2:
                tr = ord(text[0].upper()) - ord('A')
                tc = ord(text[1].upper()) - ord('A')
                if tr >= 0 and tc >= 0 and tr < 15 and tc < 15:
                    if b[tr][tc] == 0:
                        row, col = tr, tc
                        break
                    else:
                        print '已经有棋子在这里了！'
                else:
                    print text
                    print '不在棋盘内！'
            elif text.upper() == 'U':
                undo = True
                break
            elif text.upper() == 'Q':
                print b.dumps()
                return 0

        if undo == True:
            undo = False
            if len(history) == 0:
                print '棋盘已经清空，无法继续悔棋了！'
            else:
                print '悔棋中，回退历史棋局 ...'
                move = history.pop()
                b.loads(move)
        else:
            history.append(b.dumps())
            b[row][col] = 1
            b.show()

            if b.check() == 1:
                # b.show()
                print b.dumps()
                print ''
                print 'YOU WIN !!'
                return 0

            # print 'AI正在思考 ...'
            # time.sleep(0.6)
            # xtt = input('go on: ')
            # score, row, col = s.search(2, DEPTH)
            # AI（白方的输入重新）
            text_ai = num2char(seq_list[index])
            index += 1
            # 棋盘字符==>数字
            row, col = ord(text_ai[0].upper())-65, ord(text_ai[1].upper())-65
            cord = '%s%s' % (chr(ord('A') + row), chr(ord('A') + col))
            print 'AI 移动到:  %s ' % (cord)
            # xtt = input('go on: ')
            b[row][col] = 2
            xtt = input('go on:')
            b.show()

            if b.check() == 2:
                # b.show()
                print b.dumps()
                print ''
                print 'YOU LOSE.'
                return 0

    return 0


if __name__ == '__main__':
    data = read_files(sgf_home)
    x = None
    y = None
    for i in range(4800):
        y = x
        x = data.next()
        if x == None:
            print('whole loop: ', i)
            print('index: ', y['index'])
            print('index: ', y['file_name'])
            print '\n'
            break
        else:
            pass

