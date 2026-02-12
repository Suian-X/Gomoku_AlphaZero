# -*- coding: utf-8 -*-
import numpy as np


class Board(object):

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        self.states = {}
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]

    def init_board(self, start_player=0):   # 初始化棋盘
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):   # 将一维位置转换为二维坐标
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):   # 将二维坐标转换为一维位置
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def get_board_matrix(self):   # 获取当前棋盘的矩阵表示
        board = np.zeros((self.width, self.height), dtype=int)
        for move, player in self.states.items():
            x, y = move // self.width, move % self.height
            board[x, y] = player
        return board

    def find_three_threats(self, player):   # 查找形成三连威胁的位置
        board = self.get_board_matrix()
        threat_positions = np.zeros((self.width, self.height), dtype=float)
        
        for x in range(self.width):
            for y in range(self.height):
                if board[x, y] != 0:
                    continue
                    
                board[x, y] = player
                
                for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    if self.check_three_in_line(board, x, y, dx, dy, player):
                        threat_positions[x, y] = 1.0
                        break
                
                board[x, y] = 0
        
        return threat_positions

    def find_four_threats(self, player):   # 查找形成四连威胁的位置
        board = self.get_board_matrix()
        threat_positions = np.zeros((self.width, self.height), dtype=float)
        
        for x in range(self.width):
            for y in range(self.height):
                if board[x, y] != 0:
                    continue
                    
                board[x, y] = player
                
                for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    if self.check_four_in_line(board, x, y, dx, dy, player):
                        threat_positions[x, y] = 1.0
                        break
                
                board[x, y] = 0
        
        return threat_positions

    def check_three_in_line(self, board, x, y, dx, dy, player):
        count = 1
        
        for step in range(1, 5):
            nx, ny = x + step * dx, y + step * dy
            if 0 <= nx < self.width and 0 <= ny < self.height and board[nx, ny] == player:
                count += 1
            else:
                break
                
        for step in range(1, 5):
            nx, ny = x - step * dx, y - step * dy
            if 0 <= nx < self.width and 0 <= ny < self.height and board[nx, ny] == player:
                count += 1
            else:
                break
        
        if count == 3:
            return True
        
        return False

    def check_four_in_line(self, board, x, y, dx, dy, player):
        count = 1
        
        for step in range(1, 5):
            nx, ny = x + step * dx, y + step * dy
            if 0 <= nx < self.width and 0 <= ny < self.height and board[nx, ny] == player:
                count += 1
            else:
                break
                
        for step in range(1, 5):
            nx, ny = x - step * dx, y - step * dy
            if 0 <= nx < self.width and 0 <= ny < self.height and board[nx, ny] == player:
                count += 1
            else:
                break
        
        if count >= 4:
            return True
        
        return False

    def current_state(self):
        square_state = np.zeros((8, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:   # 1.0表示为当前玩家为先手方
            square_state[3][:, :] = 1.0
        
        square_state[4] = self.find_three_threats(self.current_player)
        
        opponent = 1 if self.current_player == 2 else 2
        square_state[5] = self.find_three_threats(opponent)
        
        square_state[6] = self.find_four_threats(self.current_player)
        
        square_state[7] = self.find_four_threats(opponent)
        
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
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
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if w <= width - n:
                horizontal = all(states.get(m + i, -1) == player for i in range(n))
                if horizontal:
                    return True, player

            if h <= height - n:
                vertical = all(states.get(m + i * width, -1) == player for i in range(n))
                if vertical:
                    return True, player

            if w <= width - n and h <= height - n:
                diag1 = all(states.get(m + i * (width + 1), -1) == player for i in range(n))
                if diag1:
                    return True, player

            if w >= n - 1 and h <= height - n:
                diag2 = all(states.get(m + i * (width - 1), -1) == player for i in range(n))
                if diag2:
                    return True, player

        return False, -1

    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
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

    def start_play(self, player1, player2, start_player=0, is_shown=1, temp=0.2):
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
            move = player_in_turn.get_action(self.board, temp=temp)
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
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, list(zip(states, mcts_probs, winners_z))