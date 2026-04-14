"""
Optional Minimax + Alpha-Beta Go agent.
"""

from dlgo.gotypes import Player
from dlgo.goboard import GameState, Move
from agents.policy.opening_policy import forced_center_opening_move
from agents.policy.minimax_policy import (
    default_minimax_evaluator,
    ordered_moves,
)

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

        forced_move = forced_center_opening_move(game_state)
        if forced_move is not None:
            return forced_move

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
        return default_minimax_evaluator(game_state, self._root_player)

    def _get_ordered_moves(self, game_state):
        return ordered_moves(game_state, self.max_branch)

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
