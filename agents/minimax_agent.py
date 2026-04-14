"""
Optional Minimax + Alpha-Beta Go agent.
"""

from dlgo.gotypes import Player
from dlgo import Point
from dlgo.goboard import GameState, Move
from dlgo.scoring import evaluate_territory

__all__ = ["MinimaxAgent"]


class MinimaxAgent:
    """
    Minimax agent with Alpha-Beta pruning and a light cache.
    """

    def __init__(self, max_depth=3, evaluator=None, max_branch=12):
        self.max_depth = max_depth
        self.max_branch = max_branch
        self.evaluator = evaluator or self._default_evaluator
        self.cache = GameResultCache()
        self._root_player = Player.black

    def select_move(self, game_state: GameState) -> Move:
        if game_state.is_over():
            return Move.pass_turn()

        self._root_player = game_state.next_player
        best_score = float("-inf")
        best_move = Move.pass_turn()

        for move in self._get_ordered_moves(game_state):
            next_state = game_state.apply_move(move)
            value = self.alphabeta(
                next_state,
                depth=self.max_depth - 1,
                alpha=float("-inf"),
                beta=float("inf"),
                maximizing_player=False,
            )
            if value > best_score:
                best_score = value
                best_move = move

        return best_move

    def minimax(self, game_state, depth, maximizing_player):
        terminal = self._terminal_value(game_state)
        if terminal is not None:
            return terminal
        if depth == 0:
            return self.evaluator(game_state)

        moves = self._get_ordered_moves(game_state)
        if maximizing_player:
            best = float("-inf")
            for move in moves:
                value = self.minimax(game_state.apply_move(move), depth - 1, False)
                best = max(best, value)
            return best

        best = float("inf")
        for move in moves:
            value = self.minimax(game_state.apply_move(move), depth - 1, True)
            best = min(best, value)
        return best

    def alphabeta(self, game_state, depth, alpha, beta, maximizing_player):
        terminal = self._terminal_value(game_state)
        if terminal is not None:
            return terminal
        if depth == 0:
            return self.evaluator(game_state)

        key = (game_state.next_player, game_state.board.zobrist_hash())
        cached = self.cache.get(key)
        alpha_orig = alpha
        beta_orig = beta
        if cached and cached["depth"] >= depth:
            if cached["flag"] == "exact":
                return cached["value"]
            if cached["flag"] == "lower":
                alpha = max(alpha, cached["value"])
            elif cached["flag"] == "upper":
                beta = min(beta, cached["value"])
            if alpha >= beta:
                return cached["value"]

        if maximizing_player:
            value = float("-inf")
            for move in self._get_ordered_moves(game_state):
                score = self.alphabeta(
                    game_state.apply_move(move), depth - 1, alpha, beta, False
                )
                value = max(value, score)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = float("inf")
            for move in self._get_ordered_moves(game_state):
                score = self.alphabeta(
                    game_state.apply_move(move), depth - 1, alpha, beta, True
                )
                value = min(value, score)
                beta = min(beta, value)
                if alpha >= beta:
                    break

        if value <= alpha_orig:
            flag = "upper"
        elif value >= beta_orig:
            flag = "lower"
        else:
            flag = "exact"
        self.cache.put(key, depth, value, flag)
        return value

    def _default_evaluator(self, game_state):
        territory = evaluate_territory(game_state.board)

        black_score = territory.num_black_stones + territory.num_black_territory
        white_score = territory.num_white_stones + territory.num_white_territory + 7.5
        material_margin = black_score - white_score

        # Lightweight liberty signal as tie-breaker.
        liberty_margin = 0
        board = game_state.board
        for r in range(1, board.num_rows + 1):
            for c in range(1, board.num_cols + 1):
                point = Point(r, c)
                string = board.get_go_string(point)
                if string is None:
                    continue
                if string.color == Player.black:
                    liberty_margin += string.num_liberties
                else:
                    liberty_margin -= string.num_liberties

        score = material_margin + 0.05 * liberty_margin
        if self._root_player == Player.black:
            return score
        return -score

    def _get_ordered_moves(self, game_state):
        legal = game_state.legal_moves()
        play_moves = [m for m in legal if m.is_play]
        pass_moves = [m for m in legal if m.is_pass]

        scored = []
        board = game_state.board
        center_r = (board.num_rows + 1) / 2.0
        center_c = (board.num_cols + 1) / 2.0
        player = game_state.next_player
        opponent = player.other

        for move in play_moves:
            score = 0.0
            point = move.point

            captures = 0
            for nb in point.neighbors():
                if not board.is_on_grid(nb):
                    continue
                string = board.get_go_string(nb)
                if string and string.color == opponent and string.num_liberties == 1:
                    captures += len(string.stones)
            score += captures * 5.0

            dist = abs(point.row - center_r) + abs(point.col - center_c)
            score += max(0.0, 2.5 - 0.3 * dist)

            try:
                next_state = game_state.apply_move(move)
                string = next_state.board.get_go_string(point)
                if string is not None:
                    score += min(4, string.num_liberties) * 0.5
                    if string.num_liberties == 1:
                        score -= 2.0
            except Exception:
                score -= 10.0

            scored.append((score, move))

        scored.sort(key=lambda item: item[0], reverse=True)
        ordered = [m for _, m in scored[: self.max_branch]]
        if pass_moves:
            ordered.extend(pass_moves)
        return ordered if ordered else [Move.pass_turn()]

    def _terminal_value(self, game_state):
        if not game_state.is_over():
            return None
        winner = game_state.winner()
        if winner is None:
            return 0.0
        return 10000.0 if winner == self._root_player else -10000.0


class GameResultCache:
    """
    Small transposition table.
    """

    def __init__(self):
        self.cache = {}

    def get(self, zobrist_hash):
        return self.cache.get(zobrist_hash)

    def put(self, zobrist_hash, depth, value, flag="exact"):
        current = self.cache.get(zobrist_hash)
        if current is not None and current["depth"] > depth:
            return
        self.cache[zobrist_hash] = {
            "depth": depth,
            "value": value,
            "flag": flag,
        }
