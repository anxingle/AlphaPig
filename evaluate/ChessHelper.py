from ChessBoard import ChessBoard
import random
from line_profiler import LineProfiler


def numToAlp(num):
    return chr(ord('A') + num)


def transferSymbol(sym):
    if sym == 0:
        return "."
    if sym == 1:
        return "O"
    if sym == 2:
        return "X"
    return "E"


def printBoard(chessboard):
    for i in range(chessboard.SIZE + 1):
        for j in range(chessboard.SIZE + 1):
            if i == 0 and j == 0:
                print ' ',
            elif i == 0:
                print numToAlp(j - 1),
            elif j == 0:
                print numToAlp(i - 1),
            else:
                print transferSymbol(chessboard.get_piece(i - 1, j - 1)),
        print

def printBoard2Str(chessboard):
    info_array=[]

    for i in range(chessboard.SIZE + 1):
        info_str = ""
        for j in range(chessboard.SIZE + 1):

            if i == 0 and j == 0:
                info_str+= ' '
            elif i == 0:
                info_str += numToAlp(j - 1)
            elif j == 0:
                info_str +=numToAlp(i - 1)
            else:
                info_str +=transferSymbol(chessboard.get_piece(i - 1, j - 1))
            info_str += '\t'
        # info_str +='\n'

        info_array.append(info_str)
    return info_array

def playRandomGame(chessboard):
    muser = 1
    _chess_helper_move_set = []
    for i in range(15):
        for j in range(15):
            _chess_helper_move_set.append((i, j))
    random.shuffle(_chess_helper_move_set)
    for move in _chess_helper_move_set:
        if chessboard.is_over():
            # print "No place to put."
            return -5

        r_row = move[0]
        r_col = move[1]

        return_value = chessboard.put_piece(r_row, r_col, muser)
        muser = 2 if muser == 1 else 1
        if return_value != 0:
            # print ("\n%s win one board. last move is %s %s, return value is %d" % (
            #     transferSymbol(chessboard.get_winner()), numToAlp(r_row), numToAlp(r_col), return_value))
            return return_value


if __name__ == '__main__':
    def linePro():
        lp = LineProfiler()

        # lp_wrapper = lp(cb.put_piece)
        # lp_wrapper(7, 7, 1)

        def playMuch(num):
            oi = 0
            for i in xrange(num):
                cb = ChessBoard()
                playRandomGame(cb)
                oi += cb.move_num
            print oi / num

        lp_wrapper = lp(playMuch)
        lp_wrapper(1000)
        lp.print_stats()


    def playRandom():
        cb = ChessBoard()
        return_value = playRandomGame(cb)
        printBoard(cb)
        print ("\n%s win one board. last move is %s %s, return value is %d" % (
            transferSymbol(cb.get_winner()), numToAlp(cb.get_lastmove()[0]), numToAlp(cb.get_lastmove()[1]),
            return_value))


    playRandom()
    #linePro()
