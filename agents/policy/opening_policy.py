"""
Shared opening rules for Go agents.
"""

from dlgo import Point
from dlgo.goboard import Move
from dlgo.gotypes import Player


def forced_center_opening_move(game_state):
    """
    Force the proved-optimal 5x5 black opening to the board center.

    Returns:
        Move.play(Point(3, 3)) when the rule applies, otherwise None.
    """
    board = game_state.board
    if board.num_rows != 5 or board.num_cols != 5:
        return None
    if game_state.next_player != Player.black:
        return None
    if game_state.last_move is not None:
        return None

    center = Point(3, 3)
    move = Move.play(center)
    if game_state.is_valid_move(move):
        return move
    return None
