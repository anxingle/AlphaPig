#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys, time
import random
import multiprocessing as mp


# ----------------------------------------------------------------------
# chessboard: 棋盘类，简单从字符串加载棋局或者导出字符串，判断输赢等
# ----------------------------------------------------------------------
class chessboard(object):

    def __init__(self, forbidden=0):
        # list内list
        self.__board = [[0 for n in xrange(15)] for m in xrange(15)]
        self.__forbidden = forbidden
        self.__dirs = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), \
                       (1, -1), (0, -1), (-1, -1))
        self.DIRS = self.__dirs
        self.won = {}

    # 清空棋盘
    def reset(self):
        for j in xrange(15):
            for i in xrange(15):
                self.__board[i][j] = 0
        return 0

    # 索引器
    def __getitem__(self, row):
        return self.__board[row]

    # 将棋盘转换成字符串
    def __str__(self):
        text = '  A B C D E F G H I J K L M N O\n'
        mark = ('. ', 'O ', 'X ')
        nrow = 0
        for row in self.__board:
            line = ''.join([mark[n] for n in row])
            text += chr(ord('A') + nrow) + ' ' + line
            nrow += 1
            if nrow < 15: text += '\n'
        return text

    # 转成字符串
    def __repr__(self):
        return self.__str__()

    def get(self, row, col):
        if row < 0 or row >= 15 or col < 0 or col >= 15:
            return 0
        return self.__board[row][col]

    def put(self, row, col, x):
        if row >= 0 and row < 15 and col >= 0 and col < 15:
            self.__board[row][col] = x
        return 0

    # 判断输赢，返回0（无输赢），1（白棋赢），2（黑棋赢）
    def check(self):
        board = self.__board
        dirs = ((1, -1), (1, 0), (1, 1), (0, 1))
        for i in xrange(15):
            for j in xrange(15):
                if board[i][j] == 0: continue
                # id 是该位置的棋子(0或X): i行，j列
                id = board[i][j]
                for d in dirs:
                    x, y = j, i
                    count = 0
                    for k in xrange(5):
                        if self.get(y, x) != id: break
                        y += d[0]
                        x += d[1]
                        count += 1
                    if count == 5:
                        self.won = {}
                        r, c = i, j
                        for z in xrange(5):
                            self.won[(r, c)] = 1
                            r += d[0]
                            c += d[1]
                        return id
        return 0

    # 返回数组对象
    def board(self):
        return self.__board

    # 导出棋局到字符串
    def dumps(self):
        import StringIO
        sio = StringIO.StringIO()
        board = self.__board
        for i in xrange(15):
            for j in xrange(15):
                stone = board[i][j]
                if stone != 0:
                    ti = chr(ord('A') + i)
                    tj = chr(ord('A') + j)
                    sio.write('%d:%s%s ' % (stone, ti, tj))
        return sio.getvalue()

    # 从字符串加载棋局
    def loads(self, text):
        self.reset()
        board = self.__board
        for item in text.strip('\r\n\t ').replace(',', ' ').split(' '):
            n = item.strip('\r\n\t ')
            if not n: continue
            n = n.split(':')
            stone = int(n[0])
            i = ord(n[1][0].upper()) - ord('A')
            j = ord(n[1][1].upper()) - ord('A')
            board[i][j] = stone
        return 0

    # 设置终端颜色
    def console(self, color):
        if sys.platform[:3] == 'win':
            try:
                import ctypes
            except:
                return 0
            kernel32 = ctypes.windll.LoadLibrary('kernel32.dll')
            GetStdHandle = kernel32.GetStdHandle
            SetConsoleTextAttribute = kernel32.SetConsoleTextAttribute
            GetStdHandle.argtypes = [ctypes.c_uint32]
            GetStdHandle.restype = ctypes.c_size_t
            SetConsoleTextAttribute.argtypes = [ctypes.c_size_t, ctypes.c_uint16]
            SetConsoleTextAttribute.restype = ctypes.c_long
            handle = GetStdHandle(0xfffffff5)
            if color < 0: color = 7
            result = 0
            if (color & 1): result |= 4
            if (color & 2): result |= 2
            if (color & 4): result |= 1
            if (color & 8): result |= 8
            if (color & 16): result |= 64
            if (color & 32): result |= 32
            if (color & 64): result |= 16
            if (color & 128): result |= 128
            SetConsoleTextAttribute(handle, result)
        else:
            if color >= 0:
                foreground = color & 7
                background = (color >> 4) & 7
                bold = color & 8
                sys.stdout.write(" \033[%s3%d;4%dm" % (bold and "01;" or "", foreground, background))
                sys.stdout.flush()
            else:
                sys.stdout.write(" \033[0m")
                sys.stdout.flush()
        return 0

    # 彩色输出
    def show(self):
        print '  A B C D E F G H I J K L M N O'
        mark = ('. ', 'O ', 'X ')
        nrow = 0
        self.check()
        color1 = 10
        color2 = 13
        for row in xrange(15):
            print chr(ord('A') + row),
            for col in xrange(15):
                ch = self.__board[row][col]
                if ch == 0:
                    self.console(-1)
                    print '.',
                elif ch == 1:
                    if (row, col) in self.won:
                        self.console(9)
                    else:
                        self.console(10)
                    print 'O',
                # self.console(-1)
                elif ch == 2:
                    if (row, col) in self.won:
                        self.console(9)
                    else:
                        self.console(13)
                    print 'X',
                # self.console(-1)
            self.console(-1)
            print ''
        return 0


# ----------------------------------------------------------------------
# evaluation: 棋盘评估类，给当前棋盘打分用
# ----------------------------------------------------------------------
class evaluation(object):

    def __init__(self):
        self.POS = []
        for i in xrange(15):
            row = [(7 - max(abs(i - 7), abs(j - 7))) for j in xrange(15)]
            self.POS.append(tuple(row))
        self.POS = tuple(self.POS)
        self.STWO = 1  # 冲二
        self.STHREE = 2  # 冲三
        self.SFOUR = 3  # 冲四
        self.TWO = 4  # 活二
        self.THREE = 5  # 活三
        self.FOUR = 6  # 活四
        self.FIVE = 7  # 活五
        self.DFOUR = 8  # 双四
        self.FOURT = 9  # 四三
        self.DTHREE = 10  # 双三
        self.NOTYPE = 11
        self.ANALYSED = 255  # 已经分析过
        self.TODO = 0  # 没有分析过
        self.result = [0 for i in xrange(30)]  # 保存当前直线分析值
        self.line = [0 for i in xrange(30)]  # 当前直线数据
        self.record = []  # 全盘分析结果 [row][col][方向]
        for i in xrange(15):
            self.record.append([])
            self.record[i] = []
            for j in xrange(15):
                self.record[i].append([0, 0, 0, 0])
        self.count = []  # 每种棋局的个数：count[黑棋/白棋][模式]
        for i in xrange(3):
            data = [0 for i in xrange(20)]
            self.count.append(data)
        self.reset()

    # 复位数据
    def reset(self):
        TODO = self.TODO
        count = self.count
        for i in xrange(15):
            line = self.record[i]
            for j in xrange(15):
                line[j][0] = TODO
                line[j][1] = TODO
                line[j][2] = TODO
                line[j][3] = TODO
        for i in xrange(20):
            count[0][i] = 0
            count[1][i] = 0
            count[2][i] = 0
        return 0

    # 四个方向（水平，垂直，左斜，右斜）分析评估棋盘，然后根据分析结果打分
    def evaluate(self, board, turn):
        score = self.__evaluate(board, turn)
        count = self.count
        if score < -9000:
            stone = turn == 1 and 2 or 1
            for i in xrange(20):
                if count[stone][i] > 0:
                    score -= i
        elif score > 9000:
            stone = turn == 1 and 2 or 1
            for i in xrange(20):
                if count[turn][i] > 0:
                    score += i
        return score

    # 四个方向（水平，垂直，左斜，右斜）分析评估棋盘，然后根据分析结果打分
    def __evaluate(self, board, turn):
        record, count = self.record, self.count
        TODO, ANALYSED = self.TODO, self.ANALYSED
        self.reset()
        # 四个方向分析
        for i in xrange(15):
            boardrow = board[i]
            recordrow = record[i]
            for j in xrange(15):
                if boardrow[j] != 0:
                    if recordrow[j][0] == TODO:  # 水平没有分析过？
                        self.__analysis_horizon(board, i, j)
                    if recordrow[j][1] == TODO:  # 垂直没有分析过？
                        self.__analysis_vertical(board, i, j)
                    if recordrow[j][2] == TODO:  # 左斜没有分析过？
                        self.__analysis_left(board, i, j)
                    if recordrow[j][3] == TODO:  # 右斜没有分析过
                        self.__analysis_right(board, i, j)

        FIVE, FOUR, THREE, TWO = self.FIVE, self.FOUR, self.THREE, self.TWO
        SFOUR, STHREE, STWO = self.SFOUR, self.STHREE, self.STWO
        check = {}

        # 分别对白棋黑棋计算：FIVE, FOUR, THREE, TWO等出现的次数
        for c in (FIVE, FOUR, SFOUR, THREE, STHREE, TWO, STWO):
            check[c] = 1
        for i in xrange(15):
            for j in xrange(15):
                stone = board[i][j]
                if stone != 0:
                    for k in xrange(4):
                        ch = record[i][j][k]
                        if ch in check:
                            count[stone][ch] += 1

        # 如果有五连则马上返回分数
        BLACK, WHITE = 1, 2
        if turn == WHITE:  # 当前是白棋
            if count[BLACK][FIVE]:
                return -9999
            if count[WHITE][FIVE]:
                return 9999
        else:  # 当前是黑棋
            if count[WHITE][FIVE]:
                return -9999
            if count[BLACK][FIVE]:
                return 9999

        # 如果存在两个冲四，则相当于有一个活四
        if count[WHITE][SFOUR] >= 2:
            count[WHITE][FOUR] += 1
        if count[BLACK][SFOUR] >= 2:
            count[BLACK][FOUR] += 1

        # 具体打分
        wvalue, bvalue, win = 0, 0, 0
        if turn == WHITE:
            if count[WHITE][FOUR] > 0: return 9990
            if count[WHITE][SFOUR] > 0: return 9980
            if count[BLACK][FOUR] > 0: return -9970
            if count[BLACK][SFOUR] and count[BLACK][THREE]:
                return -9960
            if count[WHITE][THREE] and count[BLACK][SFOUR] == 0:
                return 9950
            if count[BLACK][THREE] > 1 and \
                    count[WHITE][SFOUR] == 0 and \
                    count[WHITE][THREE] == 0 and \
                    count[WHITE][STHREE] == 0:
                return -9940
            if count[WHITE][THREE] > 1:
                wvalue += 2000
            elif count[WHITE][THREE]:
                wvalue += 200
            if count[BLACK][THREE] > 1:
                bvalue += 500
            elif count[BLACK][THREE]:
                bvalue += 100
            if count[WHITE][STHREE]:
                wvalue += count[WHITE][STHREE] * 10
            if count[BLACK][STHREE]:
                bvalue += count[BLACK][STHREE] * 10
            if count[WHITE][TWO]:
                wvalue += count[WHITE][TWO] * 4
            if count[BLACK][TWO]:
                bvalue += count[BLACK][TWO] * 4
            if count[WHITE][STWO]:
                wvalue += count[WHITE][STWO]
            if count[BLACK][STWO]:
                bvalue += count[BLACK][STWO]
        else:
            if count[BLACK][FOUR] > 0: return 9990
            if count[BLACK][SFOUR] > 0: return 9980
            if count[WHITE][FOUR] > 0: return -9970
            if count[WHITE][SFOUR] and count[WHITE][THREE]:
                return -9960
            if count[BLACK][THREE] and count[WHITE][SFOUR] == 0:
                return 9950
            if count[WHITE][THREE] > 1 and \
                    count[BLACK][SFOUR] == 0 and \
                    count[BLACK][THREE] == 0 and \
                    count[BLACK][STHREE] == 0:
                return -9940
            if count[BLACK][THREE] > 1:
                bvalue += 2000
            elif count[BLACK][THREE]:
                bvalue += 200
            if count[WHITE][THREE] > 1:
                wvalue += 500
            elif count[WHITE][THREE]:
                wvalue += 100
            if count[BLACK][STHREE]:
                bvalue += count[BLACK][STHREE] * 10
            if count[WHITE][STHREE]:
                wvalue += count[WHITE][STHREE] * 10
            if count[BLACK][TWO]:
                bvalue += count[BLACK][TWO] * 4
            if count[WHITE][TWO]:
                wvalue += count[WHITE][TWO] * 4
            if count[BLACK][STWO]:
                bvalue += count[BLACK][STWO]
            if count[WHITE][STWO]:
                wvalue += count[WHITE][STWO]

        # 加上位置权值，棋盘最中心点权值是7，往外一格-1，最外圈是0
        wc, bc = 0, 0
        for i in xrange(15):
            for j in xrange(15):
                stone = board[i][j]
                if stone != 0:
                    if stone == WHITE:
                        wc += self.POS[i][j]
                    else:
                        bc += self.POS[i][j]
        wvalue += wc
        bvalue += bc

        if turn == WHITE:
            return wvalue - bvalue

        return bvalue - wvalue

    # 分析横向
    def __analysis_horizon(self, board, i, j):
        line, result, record = self.line, self.result, self.record
        TODO = self.TODO
        for x in xrange(15):
            line[x] = board[i][x]
        self.analysis_line(line, result, 15, j)
        for x in xrange(15):
            if result[x] != TODO:
                record[i][x][0] = result[x]
        return record[i][j][0]

    # 分析横向
    def __analysis_vertical(self, board, i, j):
        line, result, record = self.line, self.result, self.record
        TODO = self.TODO
        for x in xrange(15):
            line[x] = board[x][j]
        self.analysis_line(line, result, 15, i)
        for x in xrange(15):
            if result[x] != TODO:
                record[x][j][1] = result[x]
        return record[i][j][1]

    # 分析左斜
    def __analysis_left(self, board, i, j):
        line, result, record = self.line, self.result, self.record
        TODO = self.TODO
        if i < j:
            x, y = j - i, 0
        else:
            x, y = 0, i - j
        k = 0
        while k < 15:
            if x + k > 14 or y + k > 14:
                break
            line[k] = board[y + k][x + k]
            k += 1
        self.analysis_line(line, result, k, j - x)
        for s in xrange(k):
            if result[s] != TODO:
                record[y + s][x + s][2] = result[s]
        return record[i][j][2]

    # 分析右斜
    def __analysis_right(self, board, i, j):
        line, result, record = self.line, self.result, self.record
        TODO = self.TODO
        if 14 - i < j:
            x, y, realnum = j - 14 + i, 14, 14 - i
        else:
            x, y, realnum = 0, i + j, j
        k = 0
        while k < 15:
            if x + k > 14 or y - k < 0:
                break
            line[k] = board[y - k][x + k]
            k += 1
        self.analysis_line(line, result, k, j - x)
        for s in xrange(k):
            if result[s] != TODO:
                record[y - s][x + s][3] = result[s]
        return record[i][j][3]

    def test(self, board):
        self.reset()
        record = self.record
        TODO = self.TODO
        for i in xrange(15):
            for j in xrange(15):
                if board[i][j] != 0 and 1:
                    if self.record[i][j][0] == TODO:
                        self.__analysis_horizon(board, i, j)
                        pass
                    if self.record[i][j][1] == TODO:
                        self.__analysis_vertical(board, i, j)
                        pass
                    if self.record[i][j][2] == TODO:
                        self.__analysis_left(board, i, j)
                        pass
                    if self.record[i][j][3] == TODO:
                        self.__analysis_right(board, i, j)
                        pass
        return 0

    # 分析一条线：五四三二等棋型
    def analysis_line(self, line, record, num, pos):
        TODO, ANALYSED = self.TODO, self.ANALYSED
        THREE, STHREE = self.THREE, self.STHREE
        FOUR, SFOUR = self.FOUR, self.SFOUR
        while len(line) < 30: line.append(0xf)
        while len(record) < 30: record.append(TODO)
        for i in xrange(num, 30):
            line[i] = 0xf
        for i in xrange(num):
            record[i] = TODO
        if num < 5:
            for i in xrange(num):
                record[i] = ANALYSED
            return 0
        stone = line[pos]
        inverse = (0, 2, 1)[stone]
        num -= 1
        xl = pos
        xr = pos
        while xl > 0:  # 探索左边界
            if line[xl - 1] != stone: break
            xl -= 1
        while xr < num:  # 探索右边界
            if line[xr + 1] != stone: break
            xr += 1
        left_range = xl
        right_range = xr
        while left_range > 0:  # 探索左边范围（非对方棋子的格子坐标）
            if line[left_range - 1] == inverse: break
            left_range -= 1
        while right_range < num:  # 探索右边范围（非对方棋子的格子坐标）
            if line[right_range + 1] == inverse: break
            right_range += 1

        # 如果该直线范围小于 5，则直接返回
        if right_range - left_range < 4:
            for k in xrange(left_range, right_range + 1):
                record[k] = ANALYSED
            return 0

        # 设置已经分析过
        for k in xrange(xl, xr + 1):
            record[k] = ANALYSED

        srange = xr - xl

        # 如果是 5连
        if srange >= 4:
            record[pos] = self.FIVE
            return self.FIVE

        # 如果是 4连
        if srange == 3:
            leftfour = False  # 是否左边是空格
            if xl > 0:
                if line[xl - 1] == 0:  # 活四
                    leftfour = True
            if xr < num:
                if line[xr + 1] == 0:
                    if leftfour:
                        record[pos] = self.FOUR  # 活四
                    else:
                        record[pos] = self.SFOUR  # 冲四
                else:
                    if leftfour:
                        record[pos] = self.SFOUR  # 冲四
            else:
                if leftfour:
                    record[pos] = self.SFOUR  # 冲四
            return record[pos]

        # 如果是 3连
        if srange == 2:  # 三连
            left3 = False  # 是否左边是空格
            if xl > 0:
                if line[xl - 1] == 0:  # 左边有气
                    if xl > 1 and line[xl - 2] == stone:
                        record[xl] = SFOUR
                        record[xl - 2] = ANALYSED
                    else:
                        left3 = True
                elif xr == num or line[xr + 1] != 0:
                    return 0
            if xr < num:
                if line[xr + 1] == 0:  # 右边有气
                    if xr < num - 1 and line[xr + 2] == stone:
                        record[xr] = SFOUR  # XXX-X 相当于冲四
                        record[xr + 2] = ANALYSED
                    elif left3:
                        record[xr] = THREE
                    else:
                        record[xr] = STHREE
                elif record[xl] == SFOUR:
                    return record[xl]
                elif left3:
                    record[pos] = STHREE
            else:
                if record[xl] == SFOUR:
                    return record[xl]
                if left3:
                    record[pos] = STHREE
            return record[pos]

        # 如果是 2连
        if srange == 1:  # 两连
            left2 = False
            if xl > 2:
                if line[xl - 1] == 0:  # 左边有气
                    if line[xl - 2] == stone:
                        if line[xl - 3] == stone:
                            record[xl - 3] = ANALYSED
                            record[xl - 2] = ANALYSED
                            record[xl] = SFOUR
                        elif line[xl - 3] == 0:
                            record[xl - 2] = ANALYSED
                            record[xl] = STHREE
                    else:
                        left2 = True
            if xr < num:
                if line[xr + 1] == 0:  # 左边有气
                    if xr < num - 2 and line[xr + 2] == stone:
                        if line[xr + 3] == stone:
                            record[xr + 3] = ANALYSED
                            record[xr + 2] = ANALYSED
                            record[xr] = SFOUR
                        elif line[xr + 3] == 0:
                            record[xr + 2] = ANALYSED
                            record[xr] = left2 and THREE or STHREE
                    else:
                        if record[xl] == SFOUR:
                            return record[xl]
                        if record[xl] == STHREE:
                            record[xl] = THREE
                            return record[xl]
                        if left2:
                            record[pos] = self.TWO
                        else:
                            record[pos] = self.STWO
                else:
                    if record[xl] == SFOUR:
                        return record[xl]
                    if left2:
                        record[pos] = self.STWO
            return record[pos]
        return 0

    def textrec(self, direction=0):
        text = []
        for i in xrange(15):
            line = ''
            for j in xrange(15):
                line += '%x ' % (self.record[i][j][direction] & 0xf)
            text.append(line)
        return '\n'.join(text)


# ----------------------------------------------------------------------
# DFS: 博弈树搜索
# ----------------------------------------------------------------------
class searcher(object):

    # 初始化
    def __init__(self):
        self.evaluator = evaluation()
        self.board = [[0 for n in xrange(15)] for i in xrange(15)]
        self.gameover = 0
        self.overvalue = 0
        self.maxdepth = 3

    # 产生当前棋局的走法
    def genmove(self, turn):
        moves = []
        board = self.board
        POSES = self.evaluator.POS
        for i in xrange(15):
            for j in xrange(15):
                if board[i][j] == 0:
                    score = POSES[i][j]
                    moves.append((score, i, j))
        moves.sort()
        moves.reverse()
        return moves

    # 递归搜索：返回最佳分数
    def __search(self, turn, depth, alpha=-0x7fffffff, beta=0x7fffffff):

        # 这里对搜索加入一定的噪音
        max_depth_value = depth
        if depth >= 2:
            max_depth_value = random.randint(1, depth)
        depth = depth if random.random() < 0.99 else max_depth_value
        # 深度为零则评估棋盘并返回
        if depth <= 0:
            score = self.evaluator.evaluate(self.board, turn)
            return score

        # 如果游戏结束则立马返回
        score = self.evaluator.evaluate(self.board, turn)
        if abs(score) >= 9999 and depth < self.maxdepth:
            return score

        # 产生新的走法
        moves = self.genmove(turn)
        bestmove = None

        # 枚举当前所有走法
        for score, row, col in moves:

            # 标记当前走法到棋盘
            self.board[row][col] = turn

            # 计算下一回合该谁走
            nturn = turn == 1 and 2 or 1

            # 深度优先搜索，返回评分，走的行和走的列
            score = - self.__search(nturn, depth - 1, -beta, -alpha)

            # 棋盘上清除当前走法
            self.board[row][col] = 0

            # 计算最好分值的走法
            # alpha/beta 剪枝
            if score > alpha:
                alpha = score
                bestmove = (row, col)
                if alpha >= beta:
                    break

        # 如果是第一层则记录最好的走法
        if depth == self.maxdepth and bestmove:
            self.bestmove = bestmove

        # 返回当前最好的分数，和该分数的对应走法
        return alpha

    # 具体搜索：传入当前是该谁走(turn=1/2)，以及搜索深度(depth)
    def search(self, turn, depth=3):
        # 这里对搜索加入一定的噪音
        max_depth_value = depth
        if depth >= 4:
            max_depth_value = random.randint(1, depth-2)
        self.maxdepth = depth if random.random() < 0.79 else max_depth_value # 0.70 的概率按照depth搜索
        self.bestmove = None
        score = self.__search(turn, depth)
        if abs(score) > 8000:
            self.maxdepth = depth if random.random() < 0.90 else max_depth_value # 0.90 的概率按照depth搜索
            score = self.__search(turn, 1)
        try:
            row, col = self.bestmove
        except Exception as e:
            print 'depth: ', depth, '  maxdepth:   ', self.maxdepth
            print 'score: ', score
            raise ValueError(("bestmove is None"))
        return score, row, col


# ----------------------------------------------------------------------
# psyco speedup
# ----------------------------------------------------------------------
def psyco_speedup():
    try:
        import psyco
        psyco.bind(chessboard)
        psyco.bind(evaluation)
    except:
        pass
    return 0


psyco_speedup()


def save_list(whole_index, sub_index, winner, choice, sgf_list):
    file_name = '/data/output/%s_%s_%s_%s_%.4f_.txt' % (whole_index, sub_index, winner, len(sgf_list), choice)
    with open(file_name, 'w') as f:
        f.write(str(sgf_list))
# ----------------------------------------------------------------------
# main game
# ----------------------------------------------------------------------
def gamemain(depth_black, depth_white, q_result, process_index, whole_index, main_process_start):
    try:
        # sub_start_time = time.time()
        sgf_list = []
        b = chessboard()
        # 黑手AI
        s_blank = searcher()
        s_blank.board = b.board()
        s = searcher()
        s.board = b.board()

        opening1 = ['1:HH 2:GI 1:II 2:HI',  '1:HH 2:GI 1:II 2:GJ',  '1:HH 2:GI 1:II 2:HJ', '1:HH 2:GI 1:II 2:IJ',  '1:HH 2:GI 1:II 2:JI',  '1:HH 2:GI 1:II 2:KH',  '1:HH 2:GI 1:II 2:JG',  '1:HH 2:GI 1:II 2:JJ',  '1:HH 2:GI 1:II 2:HG',  '1:HH 2:GI 1:II 2:GH',  '1:HH 2:GI 1:HJ 2:FI',  '1:HH 2:GI 1:HJ 2:IK',  '1:HH 2:GI 1:HJ 2:FJ',  '1:HH 2:GI 1:HJ 2:IJ',  '1:HH 2:GI 1:HJ 2:FK',  '1:HH 2:GI 1:HJ 2:JK',  '1:HH 2:GI 1:HJ 2:GJ',  '1:HH 2:GI 1:HJ 2:JJ',  '1:HH 2:GI 1:HJ 2:GK',  '1:HH 2:GI 1:HJ 2:II',  '1:HH 2:GI 1:HJ 2:HK',  '1:HH 2:GI 1:HJ 2:HG',  '1:HH 2:GI 1:HJ 2:HL',  '1:HH 2:GI 1:HJ 2:FH', '1:HH 2:GI 1:FJ 2:EI', '1:HH 2:GI 1:FJ 2:EJ', '1:HH 2:GI 1:FJ 2:EK', '1:HH 2:GI 1:FJ 2:FI', '1:HH 2:GI 1:FJ 2:FK', '1:HH 2:GI 1:FJ 2:GJ', '1:HH 2:GI 1:FJ 2:GK', '1:HH 2:GI 1:FJ 2:HI', '1:HH 2:GI 1:FJ 2:HJ', '1:HH 2:GI 1:FJ 2:HK', '1:HH 2:GI 1:FJ 2:II', '1:HH 2:GI 1:FJ 2:GG', '1:HH 2:GI 1:FJ 2:FH', '1:HH 2:GI 1:GJ 2:FH', '1:HH 2:GI 1:GJ 2:FI', '1:HH 2:GI 1:GJ 2:FJ', '1:HH 2:GI 1:GJ 2:GH', '1:HH 2:GI 1:GJ 2:GK', '1:HH 2:GI 1:GJ 2:HG', '1:HH 2:GI 1:GJ 2:HI', '1:HH 2:GI 1:GJ 2:HJ', '1:HH 2:GI 1:GJ 2:II', '1:HH 2:GI 1:IJ 2:GH', '1:HH 2:GI 1:IJ 2:FH', '1:HH 2:GI 1:IJ 2:FI', '1:HH 2:GI 1:IJ 2:GJ', '1:HH 2:GI 1:IJ 2:HJ', '1:HH 2:GI 1:IJ 2:HI', '1:HH 2:GI 1:IJ 2:II', '1:HH 2:GI 1:IJ 2:IG', '1:HH 2:GI 1:JJ 2:GH', '1:HH 2:GI 1:JJ 2:FH', '1:HH 2:GI 1:JJ 2:FI', '1:HH 2:GI 1:JJ 2:FJ', '1:HH 2:GI 1:JJ 2:FK', '1:HH 2:GI 1:JJ 2:GJ', '1:HH 2:GI 1:JJ 2:GK', '1:HH 2:GI 1:JJ 2:HI', '1:HH 2:GI 1:JJ 2:HG', '1:HH 2:GI 1:JJ 2:HJ', '1:HH 2:GI 1:JJ 2:HK', '1:HH 2:GI 1:JJ 2:IH', '1:HH 2:GI 1:JJ 2:II', '1:HH 2:GI 1:JJ 2:IJ', '1:HH 2:GI 1:HI 2:GH', '1:HH 2:GI 1:HI 2:GG', '1:HH 2:GI 1:HI 2:FH', '1:HH 2:GI 1:HI 2:FI', '1:HH 2:GI 1:HI 2:EI', '1:HH 2:GI 1:HI 2:FJ', '1:HH 2:GI 1:HI 2:FK', '1:HH 2:GI 1:HI 2:GJ', '1:HH 2:GI 1:HI 2:GK', '1:HH 2:GI 1:HI 2:HJ', '1:HH 2:GI 1:HI 2:HK', '1:HH 2:GI 1:HI 2:HG', '1:HH 2:GI 1:HI 2:IH', '1:HH 2:GI 1:HI 2:IJ', '1:HH 2:GI 1:JI 2:GG', '1:HH 2:GI 1:JI 2:GH', '1:HH 2:GI 1:JI 2:FG', '1:HH 2:GI 1:JI 2:FH', '1:HH 2:GI 1:JI 2:FI', '1:HH 2:GI 1:JI 2:GJ', '1:HH 2:GI 1:JI 2:HJ', '1:HH 2:GI 1:JI 2:IH', '1:HH 2:GI 1:JI 2:II', '1:HH 2:GI 1:JI 2:IJ', '1:HH 2:GI 1:JI 2:JH', '1:HH 2:GI 1:JI 2:JJ', '1:HH 2:GI 1:JI 2:GK']
        # opening2 = [ '1:HH 2:GI 1:IH', '1:HH 2:GI 1:JH', '1:HH 2:GI 1:IG', '1:HH 2:GI 1:JG', '1:HH 2:GH 1:FH', '1:HH 2:GH 1:FJ', '1:HH 2:GH 1:GI', '1:HH 2:GH 1:GJ', '1:HH 2:GH 1:HJ', '1:HH 2:GH 1:IH', '1:HH 2:GH 1:II', '1:HH 2:GH 1:IJ', '1:HH 2:GH 1:JH', '1:HH 2:GH 1:JI',
        # ]
        opening2 = ['1:HH 2:GI', '1:HH 2:GH']

        opening3 = [
           '1:FF 2:EG 1:DE 2:DF',
           '1:FF 2:EG 1:DE 2:DH',
           '1:FF 2:EG 1:EE 2:DF',
           '1:FF 2:EG 1:EE 2:DH',

           '1:FG 2:EH 1:DF 2:DG',
           '1:FG 2:EH 1:DF 2:DI',
           '1:FG 2:EH 1:EF 2:DG',
           '1:FG 2:EH 1:EF 2:DI',

           '1:GG 2:FH 1:EF 2:EG',
           '1:GG 2:FH 1:EF 2:EI',
           '1:GG 2:FH 1:FF 2:EG',
           '1:GG 2:FH 1:FF 2:EI',

           '1:GF 2:FG 1:EE 2:EF',
           '1:GF 2:FG 1:EE 2:EH',
           '1:GF 2:FG 1:FE 2:EF',
           '1:GF 2:FG 1:FE 2:EH']
        opening4 = ['1:FF 2:EG', '1:FG 2:EH',  '1:GF 2:FG',  '1:GG 2:FH',  '1:FF 2:EF',  '1:FG 2:EG',  '1:FH 2:EH', '1:FE 2:EE']
        choice = random.random()
        if choice <= 0.4:
            openid = random.randint(0, len(opening1) - 1)
            b.loads(opening1[openid])
            sgf_list += opening1[openid].split(' ')
        elif choice > 0.4 and choice <= 0.6:
            openid = random.randint(0, len(opening2) - 1)
            b.loads(opening2[openid])
            sgf_list += opening2[openid].split(' ')
        elif choice > 0.6 and choice < 0.85:
            openid = random.randint(0, len(opening3) - 1)
            b.loads(opening3[openid])
            sgf_list += opening3[openid].split(' ')
        else:
            openid = random.randint(0, len(opening4) - 1)
            b.loads(opening4[openid])
            sgf_list += opening4[openid].split(' ')

        turn = 2
        history = []
        undo = False

        # 设置难度
        DEPTH = depth_white
        DEPTH_BLACK = depth_black -1 if depth_black >= 4 else depth_black

        while 1:
            # print ''
            while 1:
                # print '<ROUND %d>' % (len(history) + 1)
                # 黑手AI自动下
                # text = raw_input().strip('\r\n\t ')
                score_b, tr, tc = s.search(1, DEPTH_BLACK)
                cord_b = '%s%s' % (chr(ord('A') + tr), chr(ord('A') + tc))
                sgf_cord_b = '1:%s%s' % (chr(ord('A') + tr), chr(ord('A') + tc))
                sgf_list.append(sgf_cord_b)
                if len(sgf_list) > 154 or time.time() - main_process_start.Value > 3800:
                    print('main_process:%s  sub_process:%s too long' % (whole_index, process_index))
                    print('sub process : %s takes:   %s s,   and length of history: %s' % (process_index, time.time()-main_process_start.Value, len(sgf_list) ))
                    q_result.put(-1)
                    return -1
                # print 'You move to %s   (%d)' % (cord_b, score_b)
                # if len(text) == 2:
                #     tr = ord(text[0].upper()) - ord('A')
                #     tc = ord(text[1].upper()) - ord('A')
                if tr >= 0 and tc >= 0 and tr < 15 and tc < 15:
                    if b[tr][tc] == 0:
                        row, col = tr, tc
                        break
                    else:
                        print 'can not move there'
                else:
                    print 'bad position'

            if undo == True:
                undo = False
                if len(history) == 0:
                    print 'no history to undo'
                else:
                    print 'rollback from history ...'
                    move = history.pop()
                    b.loads(move)
            else:
                history.append(b.dumps())
                b[row][col] = 1
                # b.show()

                if b.check() == 1:
                    # b.show()
                    # print b.dumps()
                    # print ''
                    # print 'YOU WIN !!'
                    print('sub process : %s takes:   %s s' % (process_index, time.time()-main_process_start.Value ))
                    save_list(whole_index, process_index, 'black', choice, sgf_list)
                    q_result.put(-1)
                    return 0

                # print 'robot is thinking now ...'
                # xtt = input('go on: ')
                score, row, col = s.search(2, DEPTH)
                # cord = '%s%s' % (chr(ord('A') + row), chr(ord('A') + col))
                sgf_cord_b_ = '2:%s%s' % (chr(ord('A') + row), chr(ord('A') + col))
                sgf_list.append(sgf_cord_b_)
                if len(sgf_list) > 154 or time.time() - main_process_start.Value > 2800:
                    print('main_process:%s  sub_process:%s too long' % (whole_index, process_index))
                    print('sub process : %s takes:   %s s,   and length of history: %s' % (process_index, time.time()-main_process_start.Value, len(sgf_list) ))
                    q_result.put(-1)
                    return -1
                # print 'robot move to %s (%d)' % (cord, score)
                # xtt = input('go on: ')
                b[row][col] = 2

                if b.check() == 2:
                    # b.show()
                    # print b.dumps()
                    # print ''
                    # print 'YOU LOSE.'
                    print('sub process : %s takes:   %s s' % (process_index, time.time()-main_process_start.Value ))
                    save_list(whole_index, process_index, 'white', choice, sgf_list)
                    q_result.put(-1)
                    return 0
        q_result.put(-1)
        return 0
    except Exception as e:
        print 'Exception:', e
        q_result.put(-1)
        return -1


def run(n_games):
    cpu_count = 10
    print('cpu count: ', cpu_count)
    for i in range(2, n_games):
        main_process_start = mp.Value("d", 0.0)
        main_process_start.Value = time.time()
        today_time = x = time.strftime("%Y-%m-%d ^%H:%M:%S", time.localtime())
        depth_black_ = random.randint(1, 2)
        depth_white_ = random.randint(1, 2)
        if depth_black_ >= depth_white_:
            depth_white_ = depth_black_ + 2
        print 'main process %s:  %s : %s ' % (i, depth_black_, depth_white_)
        # 保存多线程运行的结果
        q_result = mp.Queue()
        # 多线程任务队列
        process_list = []
        for cpu_index in range(cpu_count):
            worker = mp.Process(target=gamemain, args=(depth_black_, depth_white_, q_result, cpu_index, i, main_process_start))
            process_list.append(worker)
        for worker_index in process_list:
            worker_index.start()
        for cpu_index in range(cpu_count):
            temp_ = q_result.get()
        for worker_index in process_list:
            worker_index.join()

# ----------------------------------------------------------------------
# testing case
# ----------------------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()
    print('training.......')
    run(50)
    print('cost time：  ', time.time() - start_time)


