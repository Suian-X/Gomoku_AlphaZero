# -*- coding: utf-8 -*-
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet


if __name__ == '__main__':
    board = Board(width=9, height=9, n_in_row=5)
    game = Game(board)
    
    policy_value_net = PolicyValueNet(9, 9, model_file='best_policy.model', use_gpu=False)
    policy_value_net_best = PolicyValueNet(9, 9, model_file='more_powerful_model\\best_policy.model', use_gpu=False)
    #policy_value_net_best = PolicyValueNet(9, 9, model_file='best_policy.model', use_gpu=False)
    
    ai1 = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=5, n_playout=400, is_selfplay=0)   # 比赛模式不进行自我对弈
    ai1.set_player_ind(1)
    
    ai2 = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=5, n_playout=400, is_selfplay=0)
    ai2.set_player_ind(2)

    ai_best1 = MCTSPlayer(policy_value_net_best.policy_value_fn, c_puct=5, n_playout=400, is_selfplay=0)
    ai_best1.set_player_ind(1)

    ai_best2 = MCTSPlayer(policy_value_net_best.policy_value_fn, c_puct=5, n_playout=400, is_selfplay=0)
    ai_best2.set_player_ind(2)
    
    game.start_play(ai_best1, ai_best2, start_player=0, is_shown=1, temp=1e-3)