"""
Reusable heuristic policy helpers for the MCTS agent.
"""

from __future__ import annotations

import math
import random

from dlgo import Point, compute_game_result
from dlgo.goboard import Move
from dlgo.gotypes import Player


DEFAULT_CANDIDATE_LIMIT = 12
DEFAULT_EXPANSION_TOP_K = 4
DEFAULT_ROLLOUT_TOP_K = 6


def candidate_moves(game_state, expansion_policy="heuristic", candidate_limit=None):
    """
    Build candidate moves for a node.

    Under heuristic expansion we trim the branching factor so the same
    simulation budget can focus on stronger tactical candidates.
    """
    legal = game_state.legal_moves()
    play_moves = [move for move in legal if move.is_play]
    pass_moves = [move for move in legal if move.is_pass]

    if expansion_policy == "uniform":
        candidates = play_moves + pass_moves
        return candidates if candidates else [move for move in legal if not move.is_resign]

    limit = candidate_limit or DEFAULT_CANDIDATE_LIMIT
    scored = [(move_priority(game_state, move), move) for move in play_moves]
    scored.sort(key=lambda item: item[0], reverse=True)

    ordered = [move for _, move in scored[:limit]]
    if not ordered and pass_moves:
        return pass_moves

    if pass_moves and should_offer_pass(game_state):
        ordered.extend(pass_moves)
    return ordered if ordered else [Move.pass_turn()]


def pick_expansion_move(game_state, unexpanded_moves, expansion_policy):
    if not unexpanded_moves:
        return Move.pass_turn()
    if expansion_policy == "uniform":
        return random.choice(unexpanded_moves)

    scored = [(move_priority(game_state, move), move) for move in unexpanded_moves]
    scored.sort(key=lambda item: item[0], reverse=True)
    top_k = min(DEFAULT_EXPANSION_TOP_K, len(scored))
    top_moves = scored[:top_k]
    weights = [max(0.05, score + 3.0) for score, _ in top_moves]
    return random.choices([move for _, move in top_moves], weights=weights, k=1)[0]


def rollout_prior(game_state, move):
    if move.is_pass:
        return 0.15
    if move.is_resign:
        return 0.0
    score = move_priority(game_state, move)
    return 1.0 + max(0.0, score) / 12.0


def select_rollout_move(game_state, rollout_policy):
    legal = game_state.legal_moves()
    rollout_moves = [move for move in legal if not move.is_resign]
    if not rollout_moves:
        return Move.pass_turn()

    if rollout_policy == "random":
        return random.choice(rollout_moves)

    play_moves = [move for move in rollout_moves if move.is_play]
    if not play_moves:
        return Move.pass_turn()

    scored = [(move_priority(game_state, move), move) for move in play_moves]
    scored.sort(key=lambda item: item[0], reverse=True)

    top_pool = scored[: min(DEFAULT_ROLLOUT_TOP_K, len(scored))]
    weights = [max(0.05, score + 2.8) for score, _ in top_pool]
    chosen = random.choices([move for _, move in top_pool], weights=weights, k=1)[0]

    if should_offer_pass(game_state):
        best_score = top_pool[0][0] if top_pool else float("-inf")
        if best_score < 0.6 and random.random() < 0.25:
            return Move.pass_turn()
    return chosen


def fast_position_value(game_state, perspective_player):
    """
    Fast non-terminal evaluation used at rollout cutoffs.
    """
    result = compute_game_result(game_state)
    material_margin = result.b - (result.w + result.komi)
    shape_margin = string_feature_margin(game_state)
    atari_margin = tactical_danger_margin(game_state)

    black_eval = 0.60 * material_margin + 0.30 * shape_margin + 0.25 * atari_margin
    if perspective_player == Player.black:
        margin = black_eval
    else:
        margin = -black_eval

    return 1.0 / (1.0 + math.exp(-margin / 2.6))


def move_priority(game_state, move):
    if move.is_pass:
        return 0.2 if should_offer_pass(game_state) else -3.5
    if move.is_resign:
        return -100.0

    board = game_state.board
    point = move.point
    player = game_state.next_player
    opponent = player.other

    score = 0.0
    capture_size = 0
    save_size = 0
    friendly_neighbors = set()
    opponent_neighbors = set()
    occupied_neighbors = 0

    for neighbor in point.neighbors():
        if not board.is_on_grid(neighbor):
            continue
        string = board.get_go_string(neighbor)
        if string is None:
            continue
        occupied_neighbors += 1
        if string.color == opponent:
            opponent_neighbors.add(id(string))
            if string.num_liberties == 1:
                capture_size += len(string.stones)
        else:
            friendly_neighbors.add(id(string))
            if string.num_liberties == 1:
                save_size += len(string.stones)

    score += capture_size * 8.5
    score += save_size * 6.8

    if len(friendly_neighbors) >= 2:
        score += 2.8 * (len(friendly_neighbors) - 1)
    if len(opponent_neighbors) >= 2:
        score += 1.7 * (len(opponent_neighbors) - 1)

    center_r = (board.num_rows + 1) / 2.0
    center_c = (board.num_cols + 1) / 2.0
    dist = abs(point.row - center_r) + abs(point.col - center_c)
    score += max(0.0, 1.8 - 0.25 * dist)
    score += occupied_neighbors * 0.25

    try:
        next_state = game_state.apply_move(move)
        new_string = next_state.board.get_go_string(point)
        if new_string is not None:
            liberties = new_string.num_liberties
            score += min(5, liberties) * 0.9
            if liberties == 1:
                score -= 7.5
            elif liberties == 2:
                score -= 1.5
            elif liberties >= 4:
                score += 0.6

            score += max(0, len(new_string.stones) - 1) * 0.25

            immediate_pressure = 0
            for neighbor in point.neighbors():
                if not next_state.board.is_on_grid(neighbor):
                    continue
                neighbor_string = next_state.board.get_go_string(neighbor)
                if neighbor_string is None:
                    continue
                if (
                    neighbor_string.color == opponent
                    and neighbor_string.num_liberties == 1
                ):
                    immediate_pressure += len(neighbor_string.stones)
            score += immediate_pressure * 2.0

            if next_state.is_over():
                winner = next_state.winner()
                if winner == player:
                    score += 30.0
                elif winner == opponent:
                    score -= 30.0
    except Exception:
        score -= 20.0

    return score


def should_offer_pass(game_state):
    board_area = game_state.board.num_rows * game_state.board.num_cols
    if move_index(game_state) < board_area:
        return False
    result = compute_game_result(game_state)
    margin = result.b - (result.w + result.komi)
    if game_state.next_player == Player.white:
        margin = -margin
    return margin > 0


def tactical_danger_margin(game_state):
    margin = 0.0
    for string in iter_strings(game_state.board):
        value = 0.0
        size = len(string.stones)
        liberties = string.num_liberties
        if liberties == 1:
            value -= 3.5 + 1.3 * size
        elif liberties == 2:
            value -= 1.2 + 0.35 * size
        elif liberties >= 4:
            value += 0.5 + 0.15 * min(size, 4)

        if string.color == Player.black:
            margin += value
        else:
            margin -= value
    return margin


def string_feature_margin(game_state):
    board = game_state.board
    margin = 0.0
    for string in iter_strings(board):
        value = single_string_score(board, string)
        if string.color == Player.black:
            margin += value
        else:
            margin -= value
    return margin


def single_string_score(board, string):
    size = len(string.stones)
    liberties = string.num_liberties

    score = 0.42 * size
    score += min(5, liberties) * 0.28

    if liberties == 1:
        score -= 3.0 + 0.9 * size
    elif liberties == 2:
        score -= 0.9 + 0.25 * size
    elif liberties >= 4:
        score += 0.18 * min(size, 4)

    connection_bonus = 0.0
    for stone in string.stones:
        for neighbor in stone.neighbors():
            if not board.is_on_grid(neighbor):
                continue
            neighbor_string = board.get_go_string(neighbor)
            if neighbor_string is not None and neighbor_string.color == string.color:
                connection_bonus += 0.04
    score += min(connection_bonus, 0.8)
    return score


def iter_strings(board):
    seen = set()
    for row in range(1, board.num_rows + 1):
        for col in range(1, board.num_cols + 1):
            point = Point(row, col)
            string = board.get_go_string(point)
            if string is None:
                continue
            key = frozenset(string.stones)
            if key in seen:
                continue
            seen.add(key)
            yield string


def move_index(game_state):
    count = 0
    state = game_state
    while state is not None and state.last_move is not None:
        count += 1
        state = state.previous_state
    return count
