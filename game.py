# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np
import os
from policy_value_net_mxnet import PolicyValueNet # Keras
from mcts_alphaZero import MCTSPlayer
from utils import sgf_dataIter

current_relative_path = lambda x: os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), x))

class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.history = []
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: (histlen*2+1)*width*height
        """
        histlen = 4
        statelen = histlen*2+1
        square_state = np.zeros((statelen, self.width, self.height))
        if self.states:
            histarr = np.array(self.history)
            moves_all = histarr[:, 0]
            players_all = histarr[:, 1]
            real_len = len(moves_all)
            for i in range(histlen):
                moves = moves_all[:real_len-i]
                players = players_all[:real_len-i]
                #print(moves, players)
                move_curr = moves[players == self.current_player]
                move_oppo = moves[players != self.current_player]
                square_state[-2*i-3][move_curr // self.width,
                                move_curr % self.height] = 1.0
                square_state[-2*i-2][move_oppo // self.width,
                                move_oppo % self.height] = 1.0
                if real_len-i == 0:
                    break
        if len(self.states) % 2 == 0:
            square_state[-1][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def current_state_old(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.history.append((move, self.current_player))
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row + 2:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner


    def start_self_play(self, player, is_shown=0, temp=1e-3, sgf_home=None, file_name=None):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        # 获取棋盘数据
        X_train = sgf_dataIter.get_data_from_files(file_name, sgf_home)
        data_length = len(X_train['seq_num_list'])   # 对弈长度（一盘棋盘数据的长度）
        self.board.init_board()
        p1, p2 = self.board.players
        print('p1: ', p1, '   p2:  ', p2)
        states, mcts_probs, current_players = [], [], []
        # while True:
        for num_index, move in enumerate(X_train['seq_num_list']):
            move_, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            print('move:   ')
            print(move)
            print('move probs: ')
            # print(move_probs)
            print(type(move_probs))
            print(move_probs.shape)
            # store the data
            # print('current_state: \n')
            # print(self.board.current_state())
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            # go_on= input('go on:')
            # if go_on == 1:
            #     break
            # 既然使用现成的棋局文件， end判断当然也需要重新设置
            end, warning = 0, 1
            if num_index + 1 == data_length:
                end = 1
                winner = X_train['winner']
                try:
                    # 这是一个故意的“bug”，目的在于检验是否end
                    print('seq_num_list ...', X_train['seq_num_list'][num_index+1])
                except Exception as e:
                    # 倘若进入了这个“bug”, 则不用报告warning
                    warning = 0
                    print(e)
            # end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                # go_on = input('go on:')
                # winner 1:2
                print('winner: ', winner)
                print(X_train['file_name'])
                return warning, winner, zip(states, mcts_probs, winners_z)


if __name__ == '__main__':
    model_file = 'current_policy.model'
    policy_value_net = PolicyValueNet(15, 15)
    mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,
                                      c_puct=3,
                                      n_playout=2,
                                      is_selfplay=1)
    board = Board(width=15, height=15, n_in_row=5)
    game = Game(board)
    sgf_home = current_relative_path('./sgf_data')
    file_name = '1000_white_.sgf'
    winner, play_data = game.start_self_play(mcts_player, is_shown=1, temp=1.0, sgf_home=sgf_home, file_name=file_name)


