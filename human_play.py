# -*- coding: utf-8 -*-
from __future__ import print_function
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet


class Human(object):
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board, temp=1e-3):
        try:
            location = input("Your move (row,col): ")
            if isinstance(location, str):
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception:
            move = -1
        if move == -1 or move not in board.availables:
            print("Invalid move, try again.")
            return self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    try:
        width, height, n_in_row = 9, 9, 5
        board = Board(width=width, height=height, n_in_row=n_in_row)
        game = Game(board)

        model_file = 'more_powerful_model\\best_policy.model'
        policy_net = PolicyValueNet(width, height, model_file=model_file, use_gpu=False)
        ai_player = MCTSPlayer(policy_net.policy_value_fn,
                               c_puct=5,
                               n_playout=400,
                               is_selfplay=0)
        
        human_player = Human()

        human_first = input("Do you want to go first? (y/n): ").lower() == 'y'
        if human_first:
            human_player.set_player_ind(1)
            ai_player.set_player_ind(2)
            start_player = 0
        else:
            human_player.set_player_ind(2)
            ai_player.set_player_ind(1)
            start_player = 1

        game.start_play(human_player, ai_player, start_player=start_player, is_shown=1)

    except KeyboardInterrupt:
        print('\nQuit')


if __name__ == '__main__':
    run()