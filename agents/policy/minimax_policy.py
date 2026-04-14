"""
Reusable heuristic policy helpers for the Minimax agent.
"""

from dlgo import Point
from dlgo.goboard import Move
from dlgo.gotypes import Player
from dlgo.scoring import evaluate_territory


def default_minimax_evaluator(game_state, root_player):
    """
    Evaluate a position from the perspective of ``root_player``.
    """
    territory = evaluate_territory(game_state.board)
    komi = getattr(game_state, "komi", 7.5)

    black_score = territory.num_black_stones + territory.num_black_territory
    white_score = territory.num_white_stones + territory.num_white_territory + komi
    material_margin = black_score - white_score

    # Territory is helpful but unstable in non-terminal states, so keep it
    # as one component and add string-level tactical safety signals.
    score = 0.7 * material_margin
    score += string_feature_margin(game_state)
    if root_player == Player.black:
        return score
    return -score


def ordered_moves(game_state, max_branch):
    """
    Return legal moves ordered by the current tactical heuristic.
    """
    legal = game_state.legal_moves()
    play_moves = [move for move in legal if move.is_play]
    pass_moves = [move for move in legal if move.is_pass]

    scored = []
    board = game_state.board
    center_r = (board.num_rows + 1) / 2.0
    center_c = (board.num_cols + 1) / 2.0
    player = game_state.next_player
    opponent = player.other

    for move in play_moves:
        score = move_order_score(
            game_state,
            move,
            player=player,
            opponent=opponent,
            center_r=center_r,
            center_c=center_c,
        )
        scored.append((score, move))

    scored.sort(key=lambda item: item[0], reverse=True)
    ordered = [move for _, move in scored[:max_branch]]
    if pass_moves:
        ordered.extend(pass_moves)
    return ordered if ordered else [Move.pass_turn()]


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

    score = 0.55 * size
    score += min(5, liberties) * 0.35

    if liberties == 1:
        score -= 4.0 + 1.2 * size
    elif liberties == 2:
        score -= 1.4 + 0.35 * size
    elif liberties >= 4:
        score += 0.25 * min(size, 4)

    # Small connection / shape proxy: more shared neighbors with friendly
    # stones tends to correlate with stronger connected shapes.
    connection_bonus = 0.0
    for stone in string.stones:
        for neighbor in stone.neighbors():
            if not board.is_on_grid(neighbor):
                continue
            neighbor_string = board.get_go_string(neighbor)
            if neighbor_string is not None and neighbor_string.color == string.color:
                connection_bonus += 0.05
    score += min(connection_bonus, 1.0)

    return score


def move_order_score(game_state, move, player, opponent, center_r, center_c):
    board = game_state.board
    point = move.point
    score = 0.0

    opponent_captures = 0
    friendly_saves = 0
    friendly_neighbors = set()
    opponent_neighbors = set()

    for neighbor in point.neighbors():
        if not board.is_on_grid(neighbor):
            continue
        string = board.get_go_string(neighbor)
        if string is None:
            continue
        if string.color == opponent:
            opponent_neighbors.add(id(string))
            if string.num_liberties == 1:
                opponent_captures += len(string.stones)
        else:
            friendly_neighbors.add(id(string))
            if string.num_liberties == 1:
                friendly_saves += len(string.stones)

    score += opponent_captures * 8.0
    score += friendly_saves * 6.0

    # Connecting multiple friendly strings or touching multiple enemy
    # strings is often tactically important on small boards.
    if len(friendly_neighbors) >= 2:
        score += 2.5 * (len(friendly_neighbors) - 1)
    if len(opponent_neighbors) >= 2:
        score += 1.4 * (len(opponent_neighbors) - 1)

    dist = abs(point.row - center_r) + abs(point.col - center_c)
    score += max(0.0, 2.0 - 0.25 * dist)

    try:
        next_state = game_state.apply_move(move)
        new_string = next_state.board.get_go_string(point)
        if new_string is not None:
            liberties = new_string.num_liberties
            score += min(5, liberties) * 0.7
            if liberties == 1:
                score -= 6.0
            elif liberties == 2:
                score -= 1.0

            # Reward moves that improve a weak local group's safety.
            score += max(0, len(new_string.stones) - 1) * 0.2

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
                    score += 1.8 * len(neighbor_string.stones)
    except Exception:
        score -= 20.0

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
