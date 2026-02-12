# -*- coding: utf-8 -*-
import random
import numpy as np
import os
from collections import defaultdict, deque
from game import Board, Game
from policy_value_net import PolicyValueNet
from mcts_alphaZero import MCTSPlayer


class ReinforcementLearningTrainer():
    def __init__(self, init_model=None):
        self.board_width = 9
        self.board_height = 9
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        self.learn_rate = 5e-4
        self.lr_multiplier = 1.0
        self.temp = 0.7
        self.n_playout = 400
        self.c_puct = 5
        self.buffer_size = 50000
        self.batch_size = 512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 2000
        self.win_threshold = 0.55
        
        self.output_dir = './more_powerful_model'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if init_model:   # 加载已有模型
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model,
                                                   use_gpu=False)
        else:   # 新建网络 模型参数随机初始化
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   use_gpu=False)
        
        self.best_policy_net = PolicyValueNet(self.board_width,
                                              self.board_height,
                                              model_file=init_model,
                                              use_gpu=False)
        
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):   # 数据增强 通过旋转和翻转扩展数据集
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):   # 自我对弈 采集数据
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):   # 网络参数更新
        mini_batch = random.sample(self.data_buffer, self.batch_size)

        state_batch = np.array([data[0] for data in mini_batch], dtype=np.float32)   # 解包batch数据
        mcts_probs_batch = np.array([data[1] for data in mini_batch], dtype=np.float32)
        winner_batch = np.array([data[2] for data in mini_batch], dtype=np.float32)

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)   # 记录训练前的网络输出 用于计算KL散度
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:   # 如果KL散度过大 提前结束本轮训练
                break
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:   # 根据KL散度调整学习率
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))    # Explained Variance 衡量价值网络预测 v 与真实结果 winner 的拟合程度 当explained_var_new > explained_var_old 表示价值预测变好了
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy   # 返回总损失和策略熵

    def policy_evaluate(self, n_games=10):
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout,
                                         is_selfplay=0)
        
        best_mcts_player = MCTSPlayer(self.best_policy_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=0)
        
        win_cnt = defaultdict(int)
        for i in range(n_games):
            start_player = i % 2
            if start_player == 0:
                winner = self.game.start_play(current_mcts_player,
                                              best_mcts_player,
                                              start_player=0,
                                              is_shown=0)
            else:
                winner = self.game.start_play(best_mcts_player,
                                              current_mcts_player,
                                              start_player=0,
                                              is_shown=0)
            
            if (start_player == 0 and winner == 1) or (start_player == 1 and winner == 2):
                win_cnt[1] += 1
            elif (start_player == 0 and winner == 2) or (start_player == 1 and winner == 1):
                win_cnt[2] += 1
            else:
                win_cnt[-1] += 1
        
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("Evaluation: current model vs best model, win: {}, lose: {}, tie:{}".format(
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))

                    win_ratio = self.policy_evaluate(20)

                    self.policy_value_net.save_model('./current_policy.model')

                    if win_ratio >= self.win_threshold - 1e-3:
                        print("win_threshold: {}".format(self.win_threshold))
                        print("New best policy!!!!!!!!")
                        
                        best_model_path = os.path.join(self.output_dir, 'best_policy.model')
                        self.policy_value_net.save_model(best_model_path)
                        
                        self.best_policy_net = PolicyValueNet(self.board_width,
                                                              self.board_height,
                                                              model_file=best_model_path,
                                                              use_gpu=False)
                        
                        if self.win_threshold < 0.8:
                            self.win_threshold += 0.05
                        
                        if win_ratio == 1.0:
                            print("Current model completely defeats the best model!")
                            
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = ReinforcementLearningTrainer(init_model='best_policy.model')
    training_pipeline.run()