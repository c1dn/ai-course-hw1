"""
Policy helpers for the MCTS agent.

The current version keeps three main enhancements:
- heuristic expansion
- heuristic rollout
- static evaluation at rollout cutoffs

These helpers are tuned for 5x5 boards, where limited rollout budget
needs to focus quickly on tactically meaningful local moves.
"""

from __future__ import annotations

import math
import random

from dlgo import Point
from dlgo.goboard import Move
from dlgo.gotypes import Player

from agents.policy.minimax_policy import default_minimax_evaluator, move_order_score

# On 5x5 boards the rollout candidate pool can stay relatively wide
# without becoming prohibitively expensive.
DEFAULT_ROLLOUT_TOP_K = 16


def candidate_moves(game_state, expansion_policy="heuristic", candidate_limit=None):
    """
    Return the candidate move set for a node.

    This version avoids early pruning and keeps the full legal move set,
    leaving expansion and rollout heuristics to allocate search effort.
    """
    legal = game_state.legal_moves()
    play_moves = [move for move in legal if move.is_play]
    pass_moves = [move for move in legal if move.is_pass]
    candidates = play_moves + pass_moves
    return candidates if candidates else [move for move in legal if not move.is_resign]


def pick_expansion_move(game_state, unexpanded_moves, expansion_policy):
    """
    Select one move from the unexpanded set for node expansion.
    """
    if not unexpanded_moves:
        return Move.pass_turn()

    if expansion_policy == "heuristic":
        # Prefer high-scoring heuristic moves while keeping a little
        # randomness so tree growth does not become fully deterministic.
        scored = [(move_priority(game_state, move), move) for move in unexpanded_moves]
        scored.sort(key=lambda item: item[0], reverse=True)
        best_moves = [move for score, move in scored[:3]]
        return random.choice(best_moves)

    return random.choice(unexpanded_moves)


def rollout_prior(game_state, move):
    """
    Build a prior term from the heuristic move score.

    This nudges early tree growth toward moves that already look strong
    according to the static tactical heuristic.
    """
    score = move_priority(game_state, move)
    # Shift the score so the prior remains positive.
    return max(0.1, score + 20.0)


def select_rollout_move(game_state, rollout_policy):
    """
    Heuristic move policy used during rollout.
    """
    legal = game_state.legal_moves()
    play_moves = [move for move in legal if move.is_play]
    if not play_moves:
        return Move.pass_turn()

    if rollout_policy == "random":
        return random.choice(play_moves)

    scored = [(move_priority(game_state, move), move) for move in play_moves]
    scored.sort(key=lambda item: item[0], reverse=True)

    top_pool = scored[: min(DEFAULT_ROLLOUT_TOP_K, len(scored))]
    if not top_pool:
        return Move.pass_turn()

    # Sample from the high-scoring candidate pool using score-derived
    # weights instead of always taking the single best move.
    min_score = min(score for score, _ in top_pool)
    weights = [max(0.1, score - min_score + 1.0) for score, _ in top_pool]

    return random.choices([move for _, move in top_pool], weights=weights, k=1)[0]


def fast_position_value(game_state, perspective_player):
    """
    Fast non-terminal evaluation used at rollout cutoffs.

    This reuses the Minimax static evaluator so cutoff handling is less
    naive than plain score-difference estimation on small boards.
    """
    score = default_minimax_evaluator(game_state, perspective_player)

    # Smoothly map the evaluation margin to a probability-like value.
    return 1.0 / (1.0 + math.exp(-score / 10.0))


def move_priority(game_state, move):
    """
    Compute the heuristic move priority.

    The current score combines:
    1. Minimax-style local tactical ordering
    2. a local-reply bias for 5x5 fights
    """
    if move.is_pass:
        return -5.0
    if move.is_resign:
        return -100.0

    board = game_state.board
    center_r = (board.num_rows + 1) / 2.0
    center_c = (board.num_cols + 1) / 2.0
    player = game_state.next_player
    opponent = player.other

    # Base tactical score: captures, saves, self-atari avoidance,
    # center preference, and related local features.
    score = move_order_score(game_state, move, player, opponent, center_r, center_c)

    # Local reply bonus: on 5x5, ignoring the opponent's last move is
    # often tactically dangerous.
    if game_state.previous_state and game_state.previous_state.last_move:
        last_move = game_state.previous_state.last_move
        if last_move.is_play:
            last_point = last_move.point
            dist = abs(move.point.row - last_point.row) + abs(move.point.col - last_point.col)

            if dist == 1:
                score += 8.0
            elif dist == 2:
                score += 3.0

    return score


def move_index(game_state):
    """
    Return the number of moves already played.
    """
    count = 0
    state = game_state
    while state is not None and state.last_move is not None:
        count += 1
        state = state.previous_state
    return count
