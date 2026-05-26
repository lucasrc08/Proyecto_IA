import numpy as np
import math
import time
from collections import defaultdict

from connect4.policy import Policy


# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
ROWS, COLS       = 6, 7
WIN_REWARD       =  1.0
LOSE_REWARD      = -1.0
DRAW_REWARD      =  0.0
TIME_LIMIT = 1.0  # segundos por turno
UCB_C            = 1.41
ADP_WEIGHT       = 0.3


# ---------------------------------------------------------------------------
# Utilidades de tablero
# ---------------------------------------------------------------------------

def _get_free_cols(board):
    return [c for c in range(COLS) if board[0, c] == 0]


def _drop(board, col, player):
    nb = board.copy()
    for r in reversed(range(ROWS)):
        if nb[r, col] == 0:
            nb[r, col] = player
            break
    return nb


def _check_winner(board):
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r, c]
            if p == 0:
                continue
            if c + 3 < COLS and all(board[r, c+i] == p for i in range(4)):
                return p
            if r + 3 < ROWS and all(board[r+i, c] == p for i in range(4)):
                return p
            if r+3 < ROWS and c+3 < COLS and all(board[r+i, c+i] == p for i in range(4)):
                return p
            if r+3 < ROWS and c-3 >= 0 and all(board[r+i, c-i] == p for i in range(4)):
                return p
    return 0


def _is_final(board):
    return _check_winner(board) != 0 or len(_get_free_cols(board)) == 0


def _board_key(board):
    return board.tobytes()


# ---------------------------------------------------------------------------
# Heuristica para rollouts
# ---------------------------------------------------------------------------

def _heuristic_action(board, player):
    free = _get_free_cols(board)
    opp  = -player
    for c in free:
        if _check_winner(_drop(board, c, player)) == player:
            return c
    for c in free:
        if _check_winner(_drop(board, c, opp)) == opp:
            return c
    return sorted(free, key=lambda c: abs(c - COLS // 2))[0]


# ---------------------------------------------------------------------------
# ADP Model
# ---------------------------------------------------------------------------

class ADPModel:
    def __init__(self):
        self.N_sa = defaultdict(lambda: defaultdict(int))
        self.Q    = defaultdict(lambda: defaultdict(float))

    def update(self, s_key, a, r):
        self.N_sa[s_key][a] += 1
        n = self.N_sa[s_key][a]
        self.Q[s_key][a] += (r - self.Q[s_key][a]) / n

    def get_q(self, s_key, a):
        return self.Q[s_key][a]


# ---------------------------------------------------------------------------
# Nodo MCTS
# ---------------------------------------------------------------------------

class MCTSNode:
    def __init__(self, board, player, parent=None, action_taken=None):
        self.board        = board
        self.player       = player
        self.parent       = parent
        self.action_taken = action_taken
        self.children     = {}
        self.visits       = 0
        self.value        = 0.0
        self._untried     = _get_free_cols(board)

    def is_fully_expanded(self):
        return len(self._untried) == 0

    def is_terminal(self):
        return _is_final(self.board)

    def ucb1(self):
        if self.visits == 0:
            return float("inf")
        return (self.value / self.visits) + UCB_C * math.sqrt(
            math.log(self.parent.visits) / self.visits)

    def best_child(self):
        return max(self.children.values(), key=lambda n: n.ucb1())

    def expand(self, adp):
        col           = self._untried.pop(0)
        moving_player = -self.player
        new_board     = _drop(self.board, col, moving_player)
        child         = MCTSNode(new_board, moving_player, parent=self, action_taken=col)
        q = adp.get_q(_board_key(self.board), col)
        if q != 0.0:
            child.value  = q * ADP_WEIGHT
            child.visits = 1
        self.children[col] = child
        return child


# ---------------------------------------------------------------------------
# Motor MCTS
# ---------------------------------------------------------------------------

class MCTS:
    def __init__(self, adp):
        self.adp = adp

    def search(self, root_board, my_player, n_sim, time_limit=1.0):
        root = MCTSNode(root_board, player=-my_player)
        deadline = time.time() + time_limit
        while time.time() < deadline:
            node = self._select(root)
            if not node.is_terminal():
                node = node.expand(self.adp)
            result = self._simulate(node, my_player)
            self._backprop(node, result)
        if not root.children:
            return _heuristic_action(root_board, my_player)
        return max(root.children, key=lambda c: root.children[c].visits)

    def _select(self, node):
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            node = node.best_child()
        return node

    def _simulate(self, node, my_player):
        board  = node.board.copy()
        player = -node.player
        while not _is_final(board):
            board  = _drop(board, _heuristic_action(board, player), player)
            player = -player
        w = _check_winner(board)
        if w == my_player:  return WIN_REWARD
        if w == -my_player: return LOSE_REWARD
        return DRAW_REWARD

    def _backprop(self, node, result):
        while node is not None:
            node.visits += 1
            node.value  += result
            node = node.parent


# ---------------------------------------------------------------------------
# Politica principal
# ---------------------------------------------------------------------------

class HybridADPMCTS(Policy):

    def __init__(self, time_limit=TIME_LIMIT):
        self.time_limit   = time_limit
        self.adp          = ADPModel()
        self.mcts         = MCTS(self.adp)
        self._prev_board  = None
        self._prev_action = None
        self._my_player   = None

    def mount(self, *args, **kwargs):
        self.mcts         = MCTS(self.adp)
        self._prev_board  = None
        self._prev_action = None
        self._my_player   = None

    def act(self, s):
        board = s.copy()

        if self._my_player is None:
            neg = int(np.sum(board == -1))
            pos = int(np.sum(board == 1))
            self._my_player = -1 if neg == pos else 1

        if self._prev_board is not None and self._prev_action is not None:
            self._learn_step(self._prev_board, self._prev_action, board)

        action = self.mcts.search(board, self._my_player, n_sim=0, time_limit=self.time_limit)

        self._prev_board  = board.copy()
        self._prev_action = action

        next_board = _drop(board, action, self._my_player)
        if _is_final(next_board):
            w = _check_winner(next_board)
            r = WIN_REWARD if w == self._my_player else (
                DRAW_REWARD if w == 0 else LOSE_REWARD)
            self.adp.update(_board_key(board), action, r)

        return action

    def _learn_step(self, prev_board, action, curr_board):
        w = _check_winner(curr_board)
        if   w == self._my_player:  r = WIN_REWARD
        elif w == -self._my_player: r = LOSE_REWARD
        elif _is_final(curr_board): r = DRAW_REWARD
        else:                       r = 0.0
        self.adp.update(_board_key(prev_board), action, r)
