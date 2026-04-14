"""
MCTS (Monte Carlo Tree Search) agent for Go.
"""

import math
import random
from typing import List

from dlgo import Player, compute_game_result
from dlgo.goboard import GameState, Move

__all__ = ["MCTSAgent"]


class MCTSNode:
    """
    Search tree node used by MCTS.
    """

    def __init__(
        self,
        game_state: GameState,
        parent=None,
        move=None,
        prior=1.0,
        expansion_policy="heuristic",
        use_prior_bonus=True,
    ):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children: List["MCTSNode"] = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.expansion_policy = expansion_policy
        self.use_prior_bonus = use_prior_bonus
        self._unexpanded_moves = self._candidate_moves(game_state)

    @property
    def player(self) -> Player:
        """
        Player who made the move that led to this node.
        """
        return self.game_state.next_player.other

    @property
    def value(self) -> float:
        """Average node value."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_terminal(self) -> bool:
        return self.game_state.is_over()

    def is_fully_expanded(self) -> bool:
        return len(self._unexpanded_moves) == 0

    def best_child(self, c=1.414):
        """
        Choose child by UCT score.
        """
        best_nodes = []
        best_score = float("-inf")
        parent_visits = max(1, self.visit_count)

        for child in self.children:
            if child.visit_count == 0:
                score = float("inf")
            else:
                exploitation = child.value
                exploration = c * math.sqrt(math.log(parent_visits) / child.visit_count)
                prior_bonus = 0.0
                if self.use_prior_bonus:
                    prior_bonus = 0.15 * child.prior * math.sqrt(parent_visits) / (
                        child.visit_count + 1
                    )
                score = exploitation + exploration + prior_bonus

            if score > best_score:
                best_score = score
                best_nodes = [child]
            elif score == best_score:
                best_nodes.append(child)

        return random.choice(best_nodes) if best_nodes else None

    def expand(self):
        """
        Expand one child from unexpanded moves.
        """
        if self.is_terminal():
            return self

        if not self._unexpanded_moves:
            return random.choice(self.children) if self.children else self

        move = self._pick_expansion_move()
        self._unexpanded_moves.remove(move)
        next_state = self.game_state.apply_move(move)
        prior = self._rollout_prior(self.game_state, move)
        child = MCTSNode(
            next_state,
            parent=self,
            move=move,
            prior=prior,
            expansion_policy=self.expansion_policy,
            use_prior_bonus=self.use_prior_bonus,
        )
        self.children.append(child)
        return child

    def backup(self, value):
        """
        Backpropagate rollout value to root.

        `value` is from the perspective of this node's `player`.
        """
        node = self
        current_value = value
        while node is not None:
            node.visit_count += 1
            node.value_sum += current_value
            if current_value != 0.5:
                current_value = 1.0 - current_value
            node = node.parent

    def _pick_expansion_move(self) -> Move:
        if self.expansion_policy == "uniform":
            return random.choice(self._unexpanded_moves)

        scored = []
        for move in self._unexpanded_moves:
            scored.append((self._move_priority(self.game_state, move), move))
        scored.sort(key=lambda item: item[0], reverse=True)

        top_k = min(5, len(scored))
        if top_k == 0:
            return Move.pass_turn()
        return random.choice([m for _, m in scored[:top_k]])

    @staticmethod
    def _candidate_moves(game_state: GameState) -> List[Move]:
        legal = game_state.legal_moves()
        play_moves = [m for m in legal if m.is_play]
        pass_moves = [m for m in legal if m.is_pass]

        candidates = play_moves + pass_moves
        if not candidates:
            candidates = [m for m in legal if not m.is_resign]
        return candidates

    @staticmethod
    def _rollout_prior(game_state: GameState, move: Move) -> float:
        if move.is_pass:
            return 0.1
        if move.is_resign:
            return 0.0

        score = MCTSNode._move_priority(game_state, move)
        return 1.0 + max(0.0, score) / 10.0

    @staticmethod
    def _move_priority(game_state: GameState, move: Move) -> float:
        if move.is_pass:
            return -1.0
        if move.is_resign:
            return -100.0

        board = game_state.board
        point = move.point
        player = game_state.next_player
        opponent = player.other

        score = 0.0
        captures = 0
        for nb in point.neighbors():
            if not board.is_on_grid(nb):
                continue
            string = board.get_go_string(nb)
            if string is None:
                continue
            if string.color == opponent and string.num_liberties == 1:
                captures += len(string.stones)

        score += captures * 4.0

        center_r = (board.num_rows + 1) / 2.0
        center_c = (board.num_cols + 1) / 2.0
        dist = abs(point.row - center_r) + abs(point.col - center_c)
        score += max(0.0, 2.0 - 0.25 * dist)

        try:
            next_state = game_state.apply_move(move)
            new_string = next_state.board.get_go_string(point)
            if new_string is not None:
                score += min(3, new_string.num_liberties) * 0.5
                if new_string.num_liberties == 1:
                    score -= 1.5
        except Exception:
            score -= 5.0

        return score


class MCTSAgent:
    """
    MCTS agent implementation.
    """

    VALID_ROLLOUT_POLICIES = {"random", "heuristic"}
    VALID_EXPANSION_POLICIES = {"uniform", "heuristic"}

    def __init__(
        self,
        num_rounds=300,
        temperature=1.0,
        exploration_weight=1.414,
        max_rollout_depth=28,
        rollout_policy="heuristic",
        expansion_policy="heuristic",
        use_prior_bonus=True,
    ):
        if rollout_policy not in self.VALID_ROLLOUT_POLICIES:
            raise ValueError(
                f"Unsupported rollout_policy={rollout_policy!r}. "
                f"Expected one of {sorted(self.VALID_ROLLOUT_POLICIES)}."
            )
        if expansion_policy not in self.VALID_EXPANSION_POLICIES:
            raise ValueError(
                f"Unsupported expansion_policy={expansion_policy!r}. "
                f"Expected one of {sorted(self.VALID_EXPANSION_POLICIES)}."
            )
        self.num_rounds = num_rounds
        self.temperature = temperature
        self.exploration_weight = exploration_weight
        self.max_rollout_depth = max_rollout_depth
        self.rollout_policy = rollout_policy
        self.expansion_policy = expansion_policy
        self.use_prior_bonus = use_prior_bonus

    @classmethod
    def standard_baseline(
        cls,
        num_rounds=300,
        temperature=1.0,
        exploration_weight=1.414,
        max_rollout_depth=-1,
    ):
        """
        Construct a closer-to-textbook MCTS baseline.
        """
        return cls(
            num_rounds=num_rounds,
            temperature=temperature,
            exploration_weight=exploration_weight,
            max_rollout_depth=max_rollout_depth,
            rollout_policy="random",
            expansion_policy="uniform",
            use_prior_bonus=False,
        )

    def select_move(self, game_state: GameState) -> Move:
        if game_state.is_over():
            return Move.pass_turn()

        legal = game_state.legal_moves()
        playable = [m for m in legal if m.is_play]
        if not playable:
            return Move.pass_turn()

        root = MCTSNode(
            game_state,
            expansion_policy=self.expansion_policy,
            use_prior_bonus=self.use_prior_bonus,
        )
        rounds = self._effective_rounds(game_state)

        for _ in range(rounds):
            node = root

            while (
                not node.is_terminal()
                and node.is_fully_expanded()
                and node.children
            ):
                node = node.best_child(c=self.exploration_weight)

            if not node.is_terminal():
                node = node.expand()

            value = self._simulate(node.game_state, perspective_player=node.player)
            node.backup(value)

        return self._select_best_move(root)

    def _simulate(self, game_state, perspective_player):
        """
        Rollout with two speed optimizations:
        1) heuristic move policy
        2) rollout depth cap
        """
        state = game_state
        depth = 0

        unlimited_rollout = self.max_rollout_depth == -1
        while not state.is_over() and (unlimited_rollout or depth < self.max_rollout_depth):
            move = self._select_rollout_move(state)
            state = state.apply_move(move)
            depth += 1

        if state.is_over():
            winner = state.winner()
            if winner is None:
                return 0.5
            return 1.0 if winner == perspective_player else 0.0

        return self._fast_position_eval(state, perspective_player)

    def _select_best_move(self, root):
        if not root.children:
            return Move.pass_turn()

        scored = sorted(
            root.children,
            key=lambda child: (child.visit_count, child.value),
            reverse=True,
        )
        return scored[0].move

    def _select_rollout_move(self, game_state: GameState) -> Move:
        if self.rollout_policy == "random":
            legal = game_state.legal_moves()
            rollout_moves = [m for m in legal if not m.is_resign]
            return random.choice(rollout_moves) if rollout_moves else Move.pass_turn()

        legal = game_state.legal_moves()
        play_moves = [m for m in legal if m.is_play]
        if not play_moves:
            return Move.pass_turn()

        scored = []
        for move in play_moves:
            score = MCTSNode._move_priority(game_state, move)
            scored.append((score, move))
        scored.sort(key=lambda item: item[0], reverse=True)

        sample_pool = scored[: min(8, len(scored))]
        weights = [max(0.05, s + 1.2) for s, _ in sample_pool]
        chosen = random.choices([m for _, m in sample_pool], weights=weights, k=1)[0]

        # Light pass bias near full board to let rollouts terminate naturally.
        board_area = game_state.board.num_rows * game_state.board.num_cols
        move_index = self._move_index(game_state)
        if move_index > board_area * 1.2 and random.random() < 0.08:
            return Move.pass_turn()
        return chosen

    def _fast_position_eval(self, game_state: GameState, perspective_player: Player) -> float:
        result = compute_game_result(game_state)
        black_score = result.b
        white_score = result.w + result.komi

        if perspective_player == Player.black:
            margin = black_score - white_score
        else:
            margin = white_score - black_score

        # Smoothly map score margin to [0, 1].
        return 1.0 / (1.0 + math.exp(-margin / 3.0))

    def _effective_rounds(self, game_state: GameState) -> int:
        if self.num_rounds != -1:
            return max(1, int(self.num_rounds))
        board_area = game_state.board.num_rows * game_state.board.num_cols
        return max(320, board_area * 24)

    @staticmethod
    def _move_index(game_state: GameState) -> int:
        count = 0
        state = game_state
        while state is not None and state.last_move is not None:
            count += 1
            state = state.previous_state
        return count
