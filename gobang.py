# -*- coding: utf-8 -*-
import sys
import time
from gobang_board_utils import chessboard, evaluation, searcher, psyco_speedup


# 加速函数 
psyco_speedup()
# ----------------------------------------------------------------------
# main game
# ----------------------------------------------------------------------
def gamemain():
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
    openid = random.randint(0, len(opening) - 1)
    b.loads(opening[openid])
    turn = 2
    history = []
    undo = False

    # 设置难度
    DEPTH = 1

    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'hard':
            DEPTH = 2

    while 1:
        print ''
        while 1:
            print '<ROUND %d>' % (len(history) + 1)
            b.show()
            print '该你移动了： (u:悔棋, q:退出):',
            text = raw_input().strip('\r\n\t ')
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

            if b.check() == 1:
                b.show()
                print b.dumps()
                print ''
                print 'YOU WIN !!'
                return 0

            print 'AI正在思考 ...'
            time.sleep(1)
            # xtt = input('go on: ')
            score, row, col = s.search(2, DEPTH)
            print 'row: ', row, 'col: ', col
            cord = '%s%s' % (chr(ord('A') + row), chr(ord('A') + col))
            print 'AI 移动到:  %s 局面评分%d' % (cord, score)
            # xtt = input('go on: ')
            b[row][col] = 2

            if b.check() == 2:
                b.show()
                print b.dumps()
                print ''
                print 'YOU LOSE.'
                return 0

    return 0


# ----------------------------------------------------------------------
# testing case
# ----------------------------------------------------------------------
if __name__ == '__main__':
    gamemain()


