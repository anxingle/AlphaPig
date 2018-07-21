# -*- coding: utf-8 -*-
import requests
import cookielib
from bs4 import BeautifulSoup
import json
import time
import cPickle as pickle
import ChessHelper
from ChessBoard import ChessBoard
from Hall import GameRoom
from Hall import User
import random
import sys
import os
# 方便引入 AlphaPig
abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.insert(0, abs_path)
# import AlphaPig as gomoku_zm
import glory as gomoku_zm


class ChessClient():
    def __init__(self, server_url):
        self.session = requests.Session()
        self.session.cookies = cookielib.CookieJar()
        agent = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Maxthon/5.1.2.3000 Chrome/55.0.2883.75 Safari/537.36'
        self.headers = {
            "Host": server_url,
            "Origin": server_url,
            "Referer": server_url,
            'User-Agent': agent
        }
        self.server_url = server_url
        self.board = ChessBoard()
        self.last_status_signature = ""

    def send_get(self, url):
        return self.session.get(self.server_url + url, headers=self.headers)

    def send_post(self, url, data):
        return self.session.post(self.server_url + url, data, headers=self.headers)

    def login_in_guest(self):
        response = self.send_get('/login?action=login_in_guest')
        soup = BeautifulSoup(response.content, "html.parser")
        username_span = soup.find('span', attrs={'id': 'username'})
        if username_span:
            return username_span.text
        else:
            return None

    def login(self, username, password):
        response = self.send_post('/login?action=login',
                                  data={'username': username, 'password': password})
        soup = BeautifulSoup(response.content, "html.parser")
        username_span = soup.find('span', attrs={'id': 'username'})
        if username_span:
            return username_span.text
        else:
            return None

    def logout(self):
        self.send_get('/login?action=logout')

    def join_room(self, roomid):
        response = self.send_post('/action?action=joinroom',
                                  data={'roomid': roomid})
        action_result = json.loads(response.content)
        return action_result

    def join_game(self, cur_role):
        #response = self.send_get('/action?action=joingame')
        response = self.send_post('/action?action=joingame',
                                  data={'position': str(cur_role)})
        action_result = json.loads(response.content)
        return action_result

    def put_piece(self, row, col):
        response = self.send_get(
            '/action?action=gameaction&actionid=%s&piece_i=%d&piece_j=%d' % ('put_piece', row, col))
        action_result = json.loads(response.content)
        return action_result

    def get_room_info(self):
        response = self.send_get(
            '/action?action=gameaction&actionid=%s' % 'get_room_info')
        action_result = json.loads(response.content)
        room = pickle.loads(str(action_result['info']))
        return room

    def get_game_info(self):
        response = self.send_get(
            '/action?action=gameaction&actionid=%s' % 'get_game_info')
        action_result = json.loads(response.content)
        room = pickle.loads(str(action_result['info']))
        return room

    def get_user_info(self):
        response = self.send_get(
            '/action?action=gameaction&actionid=%s' % 'get_user_info')
        action_result = json.loads(response.content)
        room = pickle.loads(str(action_result['info']))
        return room

    def wait_game_info_changed(self, interval=0.5, max_time=100):
        wait_time = 0
        assert interval > 0, "interval must be positive"
        while True:
            response = self.send_get(
                '/action?action=gameaction&actionid=%s' % ('get_status_signature'))
            action_result = json.loads(response.content)
            if action_result['id'] == 0:
                status_signature = action_result['info']
                if self.last_status_signature != status_signature:
                    self.last_status_signature = status_signature
                    break
            else:
                print "ERROR get_status_signature,", action_result['id'], action_result['info']
                break
            time.sleep(interval)
            wait_time += interval
            if wait_time > max_time:
                break

        return wait_time

    def get_all_rooms(self):
        response = self.send_get(
            '/action?action=get_all_rooms')
        action_result = json.loads(response.content)
        all_rooms = action_result['info']
        return all_rooms

    def answer_take_back(self, agree=True):
        response = self.send_get(
            '/action?action=gameaction&actionid=answer_take_back&agree=' + ('true' if agree else 'false'))
        action_result = json.loads(response.content)
        return action_result


class GameStrategy_random():
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


class GameStrategy_MZhang():
    def __init__(self, startplayer=0, complex_='s'):
        abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
        # 普通卷积
        model_file1 = os.path.join(abs_path, './logs/best_policy_tf_10999.model')
        model_file2 = os.path.join(abs_path, './logs/current_policy_1024.model')
        model_file3 = os.path.join(abs_path, './logs/current_policy_1024.model')
        # 残差5层
        model_file_res = os.path.join(abs_path, './logs/current_res_5.model')
        policy_param = None
        self.height = 15
        self.width = 15
        if model_file1 is not None:
            print('loading...', model_file1)
            try:
                policy_param = pickle.load(open(model_file1, 'rb'))
            except:
                policy_param = pickle.load(open(model_file1, 'rb'), encoding='bytes')
        if model_file_res is not None:
            try:
                policy_param_res = pickle.load(open(model_file_res, 'rb'))
            except:
                policy_param_res = pickle.load(open(model_file_res, 'rb'), encoding='bytes')

        policy_value_net = gomoku_zm.policy_value_net_mxnet_simple.PolicyValueNet(self.height, self.width, batch_size=16, model_params=policy_param)
        # 残差网络
        policy_value_net_res = gomoku_zm.policy_value_net_mxnet_res.PolicyValueNet(self.height, self.width, batch_size=16, n_blocks=5, n_filter=128, model_params=policy_param_res)
        if complex_ == 's':
            self.mcts_player = gomoku_zm.mcts_alphaZero.MCTSPlayer(policy_value_net.policy_value_fn, c_puct=2, n_playout=300)
        elif complex_ == 'r':
            self.mcts_player = gomoku_zm.mcts_alphaZero.MCTSPlayer(policy_value_net_res.policy_value_fn, c_puct=1, n_playout=390)
        else:
            print("*=*" * 20)
            print("=*=" * 20)
            print('模型参数错误')
        self.board = gomoku_zm.game.Board(width=self.width, height=self.height, n_in_row=5)
        self.board.init_board(startplayer)
        self.game = gomoku_zm.game.Game(self.board)
        p1, p2 = self.board.players
        print('players:', p1, p2)
        self.mcts_player.set_player_ind(p1)
        pass

    def play_one_piece(self, user, gameboard):
        print('user:', gameboard.get_current_user())
        print('gameboard:', gameboard.move_history)
        lastm = gameboard.get_lastmove()
        if lastm[0] != -1:
            usr, n, row, col = lastm
            mv = (self.height-row-1)*self.height+col
            if not self.board.states.has_key(mv):
                self.board.do_move(mv)
            
        print('board:', self.board.states.items())
        move = self.mcts_player.get_action(self.board)
        print('***' * 10)
        print('move:   ', move)
        print('\n')
        self.board.do_move(move)
        # self.game.graphic(self.board, *self.board.players) 
        outmv = (self.height-move//self.height-1, move%self.width)
        
        return outmv



def go_play():
    import argparse
    import time
    parser = argparse.ArgumentParser()
    temp_room_name = 'yixin_anxingle_' + str(time.time())
    parser.add_argument('--room_name', type=str, default=temp_room_name)
    parser.add_argument('--cur_role', default=1)
    parser.add_argument('--model', default='s')
    print('room_name:  ', temp_room_name)
    parser.add_argument('--server_url', default='http://120.132.59.147:11111')
    parser.add_argument('--ai', default='random')
    args = parser.parse_args()

    client = ChessClient(args.server_url)
    client.login_in_guest()
    client.join_room(args.room_name)
    client.join_game(args.cur_role)
    user = client.get_user_info()
    print "加入游戏成功，你是:" + ("黑方" if user.game_role == 1 else "白方")
    print('model is:   ', args.model)

    if args.ai == 'random':
        strategy = GameStrategy_MZhang(user.game_role-1, args.model)
    else:
        assert False, "No other ai, you can add one or import the AICollection's ai."

    while True:
        wait_time = client.wait_game_info_changed()
        print 'wait_time:', wait_time

        room = client.get_room_info()
        # room=GameRoom()
        user = client.get_user_info()
        # user=User()
        gameboard = client.get_game_info()
        # gameboard = ChessBoard()

        print 'room.get_status():', room.get_status()
        print 'user.game_status():', user.game_status
        print 'gameboard.game_status():'
        ChessHelper.printBoard(gameboard)

        if room.get_status() == GameRoom.ROOM_STATUS_NOONE or room.get_status() == GameRoom.ROOM_STATUS_ONEWAITING:
            print "等待另一个对手加入游戏:"
            continue
        elif room.get_status() == GameRoom.ROOM_STATUS_PLAYING:
            if room.ask_take_back != 0 and room.ask_take_back != user.game_role:
                client.answer_take_back()
                break
            if gameboard.get_current_user() == user.game_role:
                print "轮到你走："
                current_time = time.time()
                one_legal_piece = strategy.play_one_piece(user, gameboard)
                action_result = client.put_piece(*one_legal_piece)
                print('***' * 20)
                print('COST TIME: %s' % (time.time() - current_time))
                if action_result['id'] != 0:
                    print "走棋失败:"
                    print ChessHelper.numToAlp(one_legal_piece[0]), ChessHelper.numToAlp(one_legal_piece[1])
                    print action_result['info']

            else:
                print "轮到对手走...."
            continue
        elif room.get_status() == GameRoom.ROOM_STATUS_FINISH:
            print "游戏已经结束了," + ("黑方" if gameboard.get_winner() == 1 else "白方") + " 赢了"
            break


if __name__ == "__main__":
    go_play()
