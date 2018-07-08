import numpy as np
import time
import cPickle as pickle


class ChessBoard(object):
    """ChessBoard

    Attributes:
        SIZE: The chess board's size.
        board: To store the board information.
        state: Indicate if the game is over.
        current_user: The user who put the next piece.
    """

    STATE_RUNNING = 0
    STATE_DONE = 1
    STATE_ABORT = 1

    PIECE_STATE_BLANK = 0
    PIECE_STATE_FIRST = 1
    PIECE_STATE_SECOND = 2

    PAD = 4

    CHECK_DIRECTION = [[[0, 1], [0, -1]], [[1, 0], [-1, 0]], [[1, 1], [-1, -1]], [[1, -1], [-1, 1]]]

    def __init__(self, size=15):
        self.SIZE = size
        self.board = np.zeros((self.SIZE + ChessBoard.PAD * 2, self.SIZE + ChessBoard.PAD * 2), dtype=np.uint8)
        self.state = ChessBoard.STATE_RUNNING
        self.current_user = ChessBoard.PIECE_STATE_FIRST

        self.move_num = 0
        self.move_history = []

        self.dump_cache = None

    def changed(func):
        def wrapper_func(self,*args, **kwargs):
            ret=func(self,*args, **kwargs)
            self.dump_cache = None
            return ret

        return wrapper_func

    def get_piece(self, row, col):
        return self.board[row + ChessBoard.PAD, col + ChessBoard.PAD]

    @changed
    def set_piece(self, row, col, user):
        self.board[row + ChessBoard.PAD, col + ChessBoard.PAD] = user

    @changed
    def put_piece(self, row, col, user):
        """Put a piece in the board and check if he wins.
        Returns:
            0 successful move.
            1 successful and win move.
            -1 move out of range.
            -2 piece has been occupied.
            -3 game is over
            -4 not your turn.
        """
        if row < 0 or row >= self.SIZE or col < 0 or col >= self.SIZE:
            return -1
        if self.get_piece(row, col) != ChessBoard.PIECE_STATE_BLANK:
            return -2
        if self.state != ChessBoard.STATE_RUNNING:
            return -3
        if user != self.current_user:
            return -4

        self.set_piece(row, col, user)
        self.move_num += 1
        self.move_history.append((user, self.move_num, row, col,))
        # self.last_move = (row, col)

        # check if win
        for dx in xrange(4):
            connected_piece_num = 1
            for dy in xrange(2):
                current_direct = ChessBoard.CHECK_DIRECTION[dx][dy]
                c_row = row
                c_col = col

                # if else realization
                for dz in xrange(4):
                    c_row += current_direct[0]
                    c_col += current_direct[1]
                    if self.get_piece(c_row, c_col) == user:
                        connected_piece_num += 1
                    else:
                        break

                # remove if, but not faster
                # p = 1
                # for dz in xrange(4):
                #     c_row += current_direct[0]
                #     c_col += current_direct[1]
                #     p = p & (self.board[c_row, c_col] == user)
                #     connected_piece_num += p

            if connected_piece_num >= 5:
                self.state = ChessBoard.STATE_DONE
                return 1

        if self.current_user == ChessBoard.PIECE_STATE_SECOND:
            self.current_user = ChessBoard.PIECE_STATE_FIRST
        else:
            self.current_user = ChessBoard.PIECE_STATE_SECOND

        if self.move_num == self.SIZE * self.SIZE:
            # self.state = ChessBoard.STATE_DONE
            self.state = ChessBoard.STATE_ABORT

        return 0

    def get_winner(self):
        return self.current_user if self.state == ChessBoard.STATE_DONE else -1

    def get_state(self):
        return self.state

    def get_current_user(self):
        return self.current_user

    def get_lastmove(self):
        return self.move_history[-1] if len(self.move_history) > 0 else (-1, -1, -1, -1)

    @changed
    def take_one_back(self):
        if len(self.move_history) > 0:
            last_move = self.move_history.pop()
            self.set_piece(last_move[-2], last_move[-1], ChessBoard.PIECE_STATE_BLANK)
            self.move_num -= 1

            if self.current_user == ChessBoard.PIECE_STATE_SECOND:
                self.current_user = ChessBoard.PIECE_STATE_FIRST
            else:
                self.current_user = ChessBoard.PIECE_STATE_SECOND

    def is_over(self):
        return self.state == ChessBoard.STATE_DONE or self.state == ChessBoard.STATE_ABORT

    def dumps(self):
        if self.dump_cache is None:
            self.dump_cache = pickle.dumps((self.SIZE, self.board, self.state, self.current_user, self.move_history))
        return self.dump_cache

    @changed
    def loads(self, chess_str):
        self.SIZE, self.board, self.state, self.current_user, self.move_history = pickle.loads(chess_str)


    @changed
    def reset(self):
        self.board = np.zeros((self.SIZE + ChessBoard.PAD * 2, self.SIZE + ChessBoard.PAD * 2), dtype=np.uint8)
        self.state = ChessBoard.STATE_RUNNING
        self.current_user = ChessBoard.PIECE_STATE_FIRST
        self.move_num = 0
        self.move_history = []

    @changed
    def abort(self):
        self.state = ChessBoard.STATE_ABORT
