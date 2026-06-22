import numpy as np
import math
import time

from connect4.policy import Policy

ROWS, COLS = 6, 7
CENTER_ORDER = sorted(range(COLS), key=lambda c: abs(c - 3))


# ─────────────────────────────────────────────
#  Funciones de tablero ultra-rápidas
# ─────────────────────────────────────────────

def _check_winner(board: np.ndarray) -> int:
    """Detección vectorizada de ganador."""
    for p in (-1, 1):
        b = board == p
        if (b[:, :-3] & b[:, 1:-2] & b[:, 2:-1] & b[:, 3:]).any(): return p
        if (b[:-3, :]  & b[1:-2, :] & b[2:-1, :] & b[3:, :]).any(): return p
        if (b[:-3,:-3] & b[1:-2,1:-2] & b[2:-1,2:-1] & b[3:,3:]).any(): return p
        if (b[:-3, 3:] & b[1:-2,2:-1] & b[2:-1,1:-2] & b[3:,:-3]).any(): return p
    return 0


def _drop(board: np.ndarray, col: int, player: int) -> int:
    """Coloca ficha in-place. Devuelve la fila."""
    row = int(np.where(board[:, col] == 0)[0][-1])
    board[row, col] = player
    return row


def _wins_at(board: np.ndarray, row: int, col: int, player: int) -> bool:
    """Verifica si hay 4 en línea pasando por (row, col). Más rápido que full scan."""
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        cnt = 1
        for s in (1, -1):
            r, c = row + dr * s, col + dc * s
            while 0 <= r < ROWS and 0 <= c < COLS and board[r, c] == player:
                cnt += 1; r += dr * s; c += dc * s
        if cnt >= 4:
            return True
    return False


def _simulate(board_orig: np.ndarray, first_mover: int, my_player: int) -> float:
    """
    Rollout completo in-place (sin copiar cada movimiento).
    Heurísticas:
      1. Jugar victoria inmediata
      2. Bloquear victoria del rival
      3. Aleatorio ponderado al centro
    Devuelve 1.0 / 0.5 / 0.0 desde la perspectiva de my_player.
    """
    board   = board_orig.copy()
    current = first_mover

    while True:
        free = [c for c in range(COLS) if board[0, c] == 0]
        if not free:
            return 0.5

        # Buscar victoria inmediata
        move = None
        for col in free:
            row = _drop(board, col, current)
            if _wins_at(board, row, col, current):
                board[row, col] = 0   # deshacer para confirmar luego
                move = col; break
            board[row, col] = 0

        # Bloquear victoria rival
        if move is None:
            for col in free:
                row = _drop(board, col, -current)
                if _wins_at(board, row, col, -current):
                    board[row, col] = 0
                    move = col; break
                board[row, col] = 0

        # Aleatorio ponderado al centro
        if move is None:
            weights = np.array([4 - abs(c - 3) for c in free], dtype=np.float32)
            weights /= weights.sum()
            move = int(np.random.choice(free, p=weights))

        row = _drop(board, move, current)
        if _wins_at(board, row, move, current):
            return 1.0 if current == my_player else 0.0
        current = -current


# ─────────────────────────────────────────────
#  Nodo del árbol MCTS
# ─────────────────────────────────────────────

class MCTSNode:
    __slots__ = ("board", "player", "parent", "action",
                 "children", "untried_actions", "wins", "visits")

    def __init__(self, board: np.ndarray, player: int,
                 parent=None, action: int = None):
        self.board   = board
        self.player  = player   # jugador que ACABA DE JUGAR
        self.parent  = parent
        self.action  = action

        self.children: list       = []
        # Columnas ordenadas centro primero
        self.untried_actions: list = [c for c in CENTER_ORDER if board[0, c] == 0]

        self.wins   = 0.0
        self.visits = 0

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child(self, c: float) -> "MCTSNode":
        log_N     = math.log(self.visits)
        best_val  = -1.0
        best_node = None
        for ch in self.children:
            if ch.visits == 0:
                return ch
            val = ch.wins / ch.visits + c * math.sqrt(log_N / ch.visits)
            if val > best_val:
                best_val  = val
                best_node = ch
        return best_node

    def expand(self) -> "MCTSNode":
        action      = self.untried_actions.pop(0)
        next_player = -self.player
        new_board   = self.board.copy()
        _drop(new_board, action, next_player)
        child = MCTSNode(new_board, next_player, parent=self, action=action)
        self.children.append(child)
        return child

    def simulate(self, my_player: int) -> float:
        first_mover = -self.player
        return _simulate(self.board, first_mover, my_player)

    def backpropagate(self, result: float) -> None:
        node = self
        while node is not None:
            node.visits += 1
            node.wins   += result
            result = 1.0 - result
            node = node.parent


# ─────────────────────────────────────────────
#  Política MCTS
# ─────────────────────────────────────────────

class MCTSPolicy(Policy):
    """
    Agente Connect-4 — Monte Carlo Tree Search con UCB1.

    Parámetros
    ----------
    time_limit   : segundos por turno (default 1.0).
    iterations   : número fijo de iteraciones (si se da, ignora time_limit).
    exploration_c: constante UCB1 (default √2).
    """

    def __init__(
        self,
        time_limit: float    = 1.0,
        iterations: int      = None,
        exploration_c: float = math.sqrt(2),
    ):
        self.time_limit    = time_limit
        self.iterations    = iterations
        self.exploration_c = exploration_c

    def mount(self, time_limit=None):
        """Gradescope pasa el timeout como argumento; lo usamos si se da."""
        if time_limit is not None:
            # Margen de seguridad: usar 80% del tiempo permitido
            self.time_limit = float(time_limit) * 0.80

    def act(self, s: np.ndarray) -> int:
        """Recibe tablero 6×7. Devuelve columna (int)."""
        act_start    = time.time()
        red_count    = int((s == -1).sum())
        yellow_count = int((s ==  1).sum())
        my_player    = -1 if red_count <= yellow_count else 1

        free = [c for c in range(COLS) if s[0, c] == 0]
        if len(free) == 1:
            return free[0]

        # ── Acciones inmediatas (antes del árbol) ──
        board_tmp = s.copy()
        for col in free:
            row = _drop(board_tmp, col, my_player)
            if _wins_at(board_tmp, row, col, my_player):
                return col          # victoria inmediata
            board_tmp[row, col] = 0

        board_tmp2 = s.copy()
        for col in free:
            row = _drop(board_tmp2, col, -my_player)
            if _wins_at(board_tmp2, row, col, -my_player):
                return col          # bloqueo inmediato
            board_tmp2[row, col] = 0

        # ── Árbol MCTS ─────────────────────────────
        root = MCTSNode(board=s.copy(), player=-my_player)

        # Hard deadline: nunca exceder time_limit desde el inicio de act()
        budget = self.time_limit if self.iterations is None else None
        if self.iterations is not None:
            for _ in range(self.iterations):
                self._iterate(root, my_player)
        else:
            deadline = act_start + budget
            while time.time() < deadline:
                self._iterate(root, my_player)

        if not root.children:
            return int(np.random.choice(free))

        best = max(root.children, key=lambda n: n.visits)
        return best.action

    def _iterate(self, root: MCTSNode, my_player: int) -> None:
        # 1. Selección
        node = root
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.exploration_c)

        # 2. Expansión
        w = _check_winner(node.board)
        if w == 0 and node.untried_actions:
            node = node.expand()

        # 3. Simulación
        result = node.simulate(my_player)

        # 4. Retropropagación
        node.backpropagate(result)
