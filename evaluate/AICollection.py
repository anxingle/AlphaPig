# -*- coding: utf-8 -*-
import time
import ChessHelper
import threading
from ChessBoard import ChessBoard
from Hall import GameRoom
from Hall import User
import random
import os
from ChessClient import ChessClient


class GameStrategy(object):
    def __init__(self):
        import gobang
        self.searcher = gobang.searcher()

    def play_one_piece(self, user, gameboard):
        # user = User()
        # gameboard = ChessBoard()
        turn = user.game_role
        self.searcher.board = [[gameboard.get_piece(m, n) for n in xrange(gameboard.SIZE)] for m in
                               xrange(gameboard.SIZE)]
        # gameboard=ChessBoard()
        # gameboard.move_history
        score, row, col = self.searcher.search(turn, 2)
        # print "score:", score
        return (row, col)


class GameStrategy_yixin(object):

    def __init__(self):
        self.muid = str(random.randint(0, 1000000))
        self.comm_folder = 'yixin_comm/'
        if not os.path.exists(self.comm_folder):
            os.makedirs(self.comm_folder)
        self.chess_state_file = self.comm_folder + 'game_state_' + self.muid
        self.action_file = self.comm_folder + 'action_' + self.muid

    def play_one_piece(self, user, gameboard):
        # user = User()
        # gameboard = ChessBoard()
        with open(self.chess_state_file, 'w') as chess_state_file:
            for userrole, move_num, row, col in gameboard.move_history:
                chess_state_file.write('%d,%d\n' % (row, col))
        if os.path.exists(self.action_file):
            os.remove(self.action_file)
        os.system("yixin_ai\yixin.exe %s %s" % (self.chess_state_file, self.action_file))
        row, col = random.randint(0, 15), random.randint(0, 15)
        with open(self.action_file) as action_file:
            line = action_file.readline()
            row, col = line.strip().split(',')
            row, col = int(row), int(col)

        return (row, col)


class GameStrategy_random(object):
    def __init__(self):
        self._chess_helper_move_set = []
        for i in range(15):
            for j in range(15):
                self._chess_helper_move_set.append((i, j))
        random.shuffle(self._chess_helper_move_set)
        self.try_step = 0

    def play_one_piece(self, user, gameboard):
        move = self._chess_helper_move_set[self.try_step]
        while gameboard.get_piece(move[0], move[1]) != 0 and self.try_step < 15 * 15:
            self.try_step += 1
            move = self._chess_helper_move_set[self.try_step]
        self.try_step += 1
        return move


class GameCommunicator(threading.Thread):
    def __init__(self, roomid, stragegy, server_url):
        threading.Thread.__init__(self)
        self.room_id = roomid
        self.stragegy = stragegy;
        self.server_url = server_url

    def run(self):
        client = ChessClient(self.server_url)
        client.login_in_guest()
        client.join_room(self.room_id)
        client.join_game()
        exp_interval = 0.5
        max_exp_interval = 200
        while True:
            single_max_time = exp_interval * 100
            wait_time = client.wait_game_info_changed(interval=exp_interval, max_time=single_max_time)
            if wait_time > single_max_time:
                exp_interval *= 2
            else:
                exp_interval = 0.5
            if exp_interval > max_exp_interval:
                exp_interval = max_exp_interval
            room = client.get_room_info()
            user = client.get_user_info()
            gameboard = client.get_game_info()
            print 'waittime:',wait_time,',room_status:',room.get_status(),',ask_take_back:',room.ask_take_back
            if room.get_status() == 1 or room.get_status() == 2:
                continue
            elif room.get_status() == 3:
                if room.ask_take_back != 0 and room.ask_take_back != user.game_role:
                    client.answer_take_back()
                    continue
                if gameboard.get_current_user() == user.game_role:
                    one_legal_piece = self.stragegy.play_one_piece(user, gameboard)
                    action_result = client.put_piece(*one_legal_piece)
                    client.wait_game_info_changed(interval=exp_interval, max_time=single_max_time)
                    if action_result['id'] != 0:
                        print ChessHelper.numToAlp(one_legal_piece[0]), ChessHelper.numToAlp(one_legal_piece[1])
                        print action_result['info']
                        break
                continue
            elif room.get_status() == 4:
                break


class GameListener(object):
    def __init__(self, prefix_stategy_map, server_url):
        self.client = ChessClient(server_url)
        self.client.login_in_guest()
        self.prefix_stategy_map = prefix_stategy_map
        self.server_url = server_url
        self.accupied = set()

    def listen(self):
        while True:
            all_rooms = self.client.get_all_rooms()
            for room in all_rooms:
                room_name = room[0]
                room_status = room[1]
                for prefix in self.prefix_stategy_map:
                    if room_name.startswith(prefix) and room_status == GameRoom.ROOM_STATUS_ONEWAITING:
                        if room_name in self.accupied:
                            continue
                        print 'Evoke:', room_name
                        strg = self.prefix_stategy_map[prefix]()
                        self.accupied.add(room_name)
                        commu = GameCommunicator(room_name, strg, self.server_url)
                        commu.start()
                        break
            time.sleep(4)


def go_listen():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_url', default='http://120.132.59.147:11111')
    args = parser.parse_args()


    if args.server_url.endswith('/'):
        args.server_url = args.server_url[:-1]
    if not args.server_url.startswith('http://'):
        args.server_url = 'http://' + args.server_url

    prefix_stategy_map = {'ai_': lambda: GameStrategy(), 'yixin_': lambda: GameStrategy_yixin(),
                          'random_': lambda: GameStrategy_random()}
    listen = GameListener(prefix_stategy_map, args.server_url)
    listen.listen()


if __name__ == "__main__":
    go_listen()
