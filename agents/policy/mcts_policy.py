"""
Reusable heuristic policy helpers for the MCTS agent.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from dlgo import Point, compute_game_result
from dlgo.goboard import Move
from dlgo.gotypes import Player
from agents.policy.minimax_policy import default_minimax_evaluator


DEFAULT_CANDIDATE_LIMIT = 12
DEFAULT_ROOT_CANDIDATE_LIMIT = 8
DEFAULT_MIDGAME_CANDIDATE_LIMIT = 6
DEFAULT_ENDGAME_CANDIDATE_LIMIT = 5
DEFAULT_EXPANSION_TOP_K = 4
DEFAULT_ROLLOUT_TOP_K = 5
DEFAULT_FORCED_MOVE_LIMIT = 4


@dataclass(frozen=True)
class MoveAnalysis:
    move: Move
    score: float
    capture_size: int
    save_size: int
    immediate_pressure: int
    new_liberties: int
    self_atari: bool


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

    limit = dynamic_candidate_limit(game_state, candidate_limit)
    analyses = score_moves(game_state, play_moves)
    ordered = compose_candidate_list(game_state, analyses, limit)
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

    analyses = score_moves(game_state, unexpanded_moves)
    forced = forced_tactical_analyses(analyses)
    if forced:
        top_forced = forced[: min(2, len(forced))]
        weights = [max(0.05, analysis.score + 4.0) for analysis in top_forced]
        return random.choices(
            [analysis.move for analysis in top_forced],
            weights=weights,
            k=1,
        )[0]

    top_k = min(DEFAULT_EXPANSION_TOP_K, len(analyses))
    top_moves = analyses[:top_k]
    weights = [max(0.05, analysis.score + 3.0) for analysis in top_moves]
    return random.choices(
        [analysis.move for analysis in top_moves],
        weights=weights,
        k=1,
    )[0]


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

    analyses = score_moves(game_state, play_moves)
    forced = forced_tactical_analyses(analyses)
    if forced:
        return forced[0].move

    top_pool = analyses[: min(DEFAULT_ROLLOUT_TOP_K, len(analyses))]
    if not top_pool:
        return Move.pass_turn()

    # Early rollout steps are more reliable when we bias harder toward the
    # strongest tactical move instead of adding extra sampling noise.
    if len(top_pool) == 1 or move_index(game_state) < 8 or random.random() < 0.60:
        chosen = top_pool[0].move
    else:
        weights = [max(0.05, analysis.score + 2.8) for analysis in top_pool]
        chosen = random.choices(
            [analysis.move for analysis in top_pool],
            weights=weights,
            k=1,
        )[0]

    if should_offer_pass(game_state):
        best_score = top_pool[0].score if top_pool else float("-inf")
        if best_score < 0.6 and random.random() < 0.25:
            return Move.pass_turn()
    return chosen


def fast_position_value(game_state, perspective_player):
    """
    Fast non-terminal evaluation used at rollout cutoffs.
    """
    black_eval = default_minimax_evaluator(game_state, Player.black)
    atari_margin = tactical_danger_margin(game_state)
    black_eval += 0.18 * atari_margin

    if perspective_player == Player.black:
        margin = black_eval
    else:
        margin = -black_eval

    return 1.0 / (1.0 + math.exp(-margin / 3.2))


def move_priority(game_state, move):
    return analyze_move(game_state, move).score


def score_moves(game_state, moves):
    analyses = [analyze_move(game_state, move) for move in moves]
    analyses.sort(key=lambda item: item.score, reverse=True)
    return analyses


def analyze_move(game_state, move):
    if move.is_pass:
        return MoveAnalysis(
            move=move,
            score=0.2 if should_offer_pass(game_state) else -3.5,
            capture_size=0,
            save_size=0,
            immediate_pressure=0,
            new_liberties=0,
            self_atari=False,
        )
    if move.is_resign:
        return MoveAnalysis(
            move=move,
            score=-100.0,
            capture_size=0,
            save_size=0,
            immediate_pressure=0,
            new_liberties=0,
            self_atari=False,
        )

    board = game_state.board
    point = move.point
    player = game_state.next_player
    opponent = player.other

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

    score = 0.0
    score += capture_size * 9.0
    score += save_size * 7.0

    if len(friendly_neighbors) >= 2:
        score += 2.8 * (len(friendly_neighbors) - 1)
    if len(opponent_neighbors) >= 2:
        score += 1.7 * (len(opponent_neighbors) - 1)

    center_r = (board.num_rows + 1) / 2.0
    center_c = (board.num_cols + 1) / 2.0
    dist = abs(point.row - center_r) + abs(point.col - center_c)
    score += max(0.0, 1.8 - 0.25 * dist)
    score += occupied_neighbors * 0.20

    immediate_pressure = 0
    new_liberties = 0
    self_atari = False

    try:
        next_state = game_state.apply_move(move)
        new_string = next_state.board.get_go_string(point)
        if new_string is not None:
            new_liberties = new_string.num_liberties
            score += min(5, new_liberties) * 0.9
            if new_liberties == 1:
                score -= 7.5
                self_atari = capture_size == 0 and save_size == 0
            elif new_liberties == 2:
                score -= 1.5
            elif new_liberties >= 4:
                score += 0.6

            score += max(0, len(new_string.stones) - 1) * 0.25

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

    return MoveAnalysis(
        move=move,
        score=score,
        capture_size=capture_size,
        save_size=save_size,
        immediate_pressure=immediate_pressure,
        new_liberties=new_liberties,
        self_atari=self_atari,
    )


def dynamic_candidate_limit(game_state, candidate_limit):
    board = game_state.board
    board_area = board.num_rows * board.num_cols
    empties = count_empty_points(board)

    if move_index(game_state) <= 2:
        dynamic_limit = DEFAULT_ROOT_CANDIDATE_LIMIT
    elif empties > board_area * 0.45:
        dynamic_limit = DEFAULT_MIDGAME_CANDIDATE_LIMIT
    else:
        dynamic_limit = DEFAULT_ENDGAME_CANDIDATE_LIMIT

    upper_bound = candidate_limit or DEFAULT_CANDIDATE_LIMIT
    return max(1, min(dynamic_limit, upper_bound))


def compose_candidate_list(game_state, analyses, limit):
    ordered = []
    seen = set()

    for analysis in forced_tactical_analyses(analyses):
        key = move_identity(analysis.move)
        if key in seen:
            continue
        ordered.append(analysis.move)
        seen.add(key)
        if len(ordered) >= limit:
            return ordered

    for analysis in analyses:
        key = move_identity(analysis.move)
        if key in seen:
            continue
        ordered.append(analysis.move)
        seen.add(key)
        if len(ordered) >= limit:
            break
    return ordered


def forced_tactical_analyses(analyses):
    forced = []
    for analysis in analyses:
        if analysis.capture_size > 0:
            forced.append(analysis)
            continue
        if analysis.save_size > 0:
            forced.append(analysis)
            continue
        if analysis.immediate_pressure >= 2 and not analysis.self_atari:
            forced.append(analysis)
            continue
    return forced[:DEFAULT_FORCED_MOVE_LIMIT]


def move_identity(move):
    if move.is_pass:
        return ("pass",)
    if move.is_resign:
        return ("resign",)
    return ("play", move.point.row, move.point.col)


def count_empty_points(board):
    empty = 0
    for row in range(1, board.num_rows + 1):
        for col in range(1, board.num_cols + 1):
            if board.get(Point(row, col)) is None:
                empty += 1
    return empty


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
