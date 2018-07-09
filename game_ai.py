# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np
from game import Board
import random

class Game_AI(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board
        self._boardSize = board.width * board.height

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

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        blank_move_list = [0,1,2,3,4,5,6,7,8,15,16,17,18,19,20,21,22,23,30,31,32,33,34,35,36,37,38,45,46,47,48,49,50,51,52,53,60,61,62,63,64,65,66,67,68,75,76,77,78,79,80,81,82,83,90,91,92,93,94,95,96,97,98]
        white_move_list = range(0, 103)
        if random.random() < 0.15:
            while True:
                move_blank = random.choice(blank_move_list)
                # move_blank = blank_move_list[random.randint(0, len()-1)]
                move_white = random.choice(white_move_list)
                if move_blank != move_white:
                    break
            # store the data
            # 黑子走子概率
            probs = [0.000001 for _ in range(self._boardSize)]
            probs[move_blank] = 0.99999
            move_blank_probs = np.asarray(probs)

            states.append(self.board.current_state())
            mcts_probs.append(move_blank_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move_blank)
            if is_shown:
                self.graphic(self.board, p1, p2)

            # 白子走子概率
            probs_ = [0.000001 for _ in range(self._boardSize)]
            probs_[move_white] = 0.99999
            move_white_probs = np.asarray(probs_)

            states.append(self.board.current_state())
            mcts_probs.append(move_white_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move_white)
            if is_shown:
                self.graphic(self.board, p1, p2)

        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
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
                return winner, zip(states, mcts_probs, winners_z)
