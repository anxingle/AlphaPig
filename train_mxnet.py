# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
         anxingle
"""

from __future__ import print_function
import pickle
import random
import os
import time
import numpy as np
from optparse import OptionParser
import multiprocessing as mp
from collections import defaultdict, deque
from game import Board, Game
from game_ai import Game_AI
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from utils import config_loader, send_email

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet # Keras
from policy_value_net_mxnet import PolicyValueNet # Mxnet

import logging
import logging.config
logging.config.dictConfig(config_loader.config_['train_logging'])
_logger = logging.getLogger(__name__)

current_relative_path = lambda x: os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), x))

class TrainPipeline():
    def __init__(self, conf, init_model=None):
        # params of the board and the game
        self.board_width = conf['board_width']
        self.board_height = conf['board_height']
        self.n_in_row = conf['n_in_row']
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        self.game_ai = Game_AI(self.board)
        # training params
        self.learn_rate = conf['learn_rate']
        self.lr_multiplier = conf['lr_multiplier']  # adaptively adjust the learning rate based on KL
        self.temp = conf['temp']  # the temperature param
        self.n_playout = conf['n_playout'] # 500  # num of simulations for each move
        self.c_puct = conf['c_puct']
        self.buffer_size = conf['buffer_size']
        self.batch_size = conf['batch_size'] # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = conf['play_batch_size']
        self.epochs = conf['epochs']  # num of train_steps for each update
        self.kl_targ = conf['kl_targ']
        self.check_freq = conf['check_freq']
        self.game_batch_num =conf['game_batch_num']
        self.best_win_ratio = 0.0
        # 多线程相关
        self._cpu_count = mp.cpu_count() - 8
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = conf['pure_mcts_playout_num']
        # 训练集文件
        self._sgf_home = current_relative_path(conf['sgf_dir'])
        _logger.info('path: %s' % self._sgf_home)
        self._ai_data_home = current_relative_path(conf['ai_data_dir'])
        # 加载人类对弈数据
        self._load_training_data(self._sgf_home)
        # 加载保存的自对弈数据
        self._load_pickle_data(self._ai_data_home)
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   self.batch_size,
                                                   n_blocks=10,
                                                   n_filter=128,
                                                   model_params=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   self.batch_size,
                                                   n_blocks=10,
                                                   n_filter=128)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def _load_training_data(self, data_dir):
        file_list = os.listdir(data_dir)
        self._training_data = [item for item in file_list if item.endswith('.sgf') and os.path.isfile(os.path.join(data_dir, item))]
        random.shuffle(self._training_data)
        self._length_train_data = len(self._training_data)

    def _load_pickle_data(self, data_dir):
        file_list = os.listdir(data_dir)
        txt_list = [item for item in file_list if item.endswith('.txt') and os.path.isfile(os.path.join(data_dir, item))]
        self._ai_history_data = []
        for txt_f in txt_list:
            f_object = open(os.path.join(data_dir, txt_f), 'rb')
            d = pickle.load(f_object)
            self._ai_history_data += d
            f_object.close()

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1, training_index=None):
        """collect SGF file data for training"""
        data_index = training_index % self._length_train_data
        if data_index == 0:
            random.shuffle(self._training_data)
        for i in range(n_games):
            warning, winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp, sgf_home=self._sgf_home, file_name=self._training_data[data_index])
            if warning:
                _logger.error('\033[0;41m %s \033[0m anxingle_training_index: %s, data_index: %s, file: %s' % ('WARNING', training_index, data_index, self._training_data[data_index]))
            else:
                _logger.info('winner: %s, file: %s ' % (winner, self._training_data[data_index]))
                # print('play_data:  ', play_data)
                play_data = list(play_data)[:]
                self.episode_len = len(play_data)
                # augment the data
                play_data = self.get_equi_data(play_data)
                self.data_buffer.extend(play_data)
        _logger.info('game_batch_index: %s, length of data_buffer: %s' % (training_index, len(self.data_buffer)))

    def collect_selfplay_data_pickle(self, n_games=1, training_index=None):
        """ 使用pickle保存的曾经的对弈数据 """
        data_index = training_index % len(self._ai_history_data)
        if data_index == 0:
            random.shuffle(self._ai_history_data)
        for i in range(n_games):
            play_data = self._ai_history_data[data_index]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def collect_selfplay_data_ai(self, n_games=1, training_index=None):
        """collect AI self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game_ai.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            _logger.info('traing_index: %s,   winner is: %s' % (training_index, winner))
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
        #print(len(self.data_buffer), n_games)

    # def _multiprocess_collect_selfplay_data(self, q, process_index):
    #     """
    #     多进程自对弈收集数据
    #     Args:
    #         q: 队列，保存结果
    #     """
    #     winner, play_data = self.game.start_self_play(self.mcts_player,
    #                                                       temp=self.temp)
    #     play_data = list(play_data)[:]
    #     self.episode_len = len(play_data)
    #     # augment the data
    #     play_data = self.get_equi_data(play_data)
    #     q.put(play_data)


    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        learn_rate = self.learn_rate*self.lr_multiplier
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    learn_rate)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                _logger.info('early stopping. i:%s.   epochs: %s' % (i, self.epochs))
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.05:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 20:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        _logger.info(("kl:{:.4f},"
               "lr:{:.1e},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        learn_rate,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        _logger.info("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                current_time = time.time()
                if i < 2000:
                    self.collect_selfplay_data_pickle(1, training_index=i)
                elif i < 7000:
                    self.collect_selfplay_data(1, training_index=i)
                elif i < 12000:
                    if i%8 == 0 or i%8 == 4:
                        self.collect_selfplay_data_pickle(1, training_index=i)
                    elif i % 8 == 3:
                        self.collect_selfplay_data(1, training_index=i)
                    else:
                        self.collect_selfplay_data_ai(1, training_index=i)
                else:
                    self.collect_selfplay_data_ai(1, training_index=i)
                _logger.info('collection cost time: %d ' % (time.time() - current_time))
                _logger.info("batch i:{}, episode_len:{}, buffer_len:{}".format(
                        i+1, self.episode_len, len(self.data_buffer)))
                if len(self.data_buffer) > self.batch_size:
                    batch_time = time.time()
                    loss, entropy = self.policy_update()
                    _logger.info('train batch cost time: %d' % (time.time() - batch_time))
                # check the performance of the current model,
                # and save the model params
                if (i+1) % 50 == 0:
                    self.policy_value_net.save_model('./logs/current_policy.model')
                if (i+1) % self.check_freq == 0:
                    check_time = time.time()
                    _logger.info("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    _logger.info('evaluate the network cost time: %s ', int(time.time() - check_time))
                    if win_ratio > self.best_win_ratio:
                        _logger.info("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./logs/best_policy_%s.model' % i)
                        if (self.best_win_ratio >= 0.98 and
                                self.pure_mcts_playout_num < 8000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            _logger.info('\n\rquit')


if __name__ == '__main__':
    try:
        start_time = time.time()
        # model_file = './logs/current_policy.model'
        model_file = None
        policy_param = None 
        conf = config_loader.load_config('./conf/train_config.yaml')
        if model_file is not None:
            _logger.info('loading...%s' %  model_file)
            try:
                policy_param = pickle.load(open(model_file, 'rb'))
            except:
                policy_param = pickle.load(open(model_file, 'rb'),
                                           encoding='bytes')  # To support python3
        training_pipeline = TrainPipeline(conf, policy_param)
        _logger.info('enter training!')
        # training_pipeline.collect_selfplay_data(1, 1)
        training_pipeline.run()
    except Exception as e:
        _logger.exception(e)
    finally:
        cost_time = int(time.time() - start_time)
        format_time = "耗时： %s 小时 %s 分 %s 秒" % (cost_time/3600, (cost_time%3600)/60, (cost_time%3600)%60 )
        send_email.send_mail('训练结束', format_time, 'XXX')
