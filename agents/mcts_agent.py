"""
MCTS (Monte Carlo Tree Search) agent for Go.

Implements the standard MCTS loop and augments it with heuristic
expansion, heuristic rollouts, prior bonus guidance, and rollout
depth limits for stronger play on 5x5 boards.
"""

import math
import random
from typing import List

from dlgo import Player
from dlgo.goboard import GameState, Move
from agents.policy.opening_policy import forced_center_opening_move
from agents.policy.mcts_policy import (
    candidate_moves,
    fast_position_value,
    move_index,
    move_priority,
    pick_expansion_move,
    rollout_prior,
    select_rollout_move,
)

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
        candidate_limit=None,
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
        self.candidate_limit = candidate_limit
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
        Choose a child using UCT plus optional prior bonus.
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
        Expand one child from the unexpanded move set.
        """
        if self.is_terminal():
            return self

        if not self._unexpanded_moves:
            return random.choice(self.children) if self.children else self

        move = self._pick_expansion_move()
        self._unexpanded_moves.remove(move)
        next_state = self.game_state.apply_move(move)
        prior = rollout_prior(self.game_state, move)
        child = MCTSNode(
            next_state,
            parent=self,
            move=move,
            prior=prior,
            expansion_policy=self.expansion_policy,
            use_prior_bonus=self.use_prior_bonus,
            candidate_limit=self.candidate_limit,
        )
        self.children.append(child)
        return child

    def backup(self, value):
        """
        Backpropagate rollout value to the root.

        ``value`` is from the perspective of this node's ``player``.
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
        return pick_expansion_move(
            self.game_state,
            self._unexpanded_moves,
            self.expansion_policy,
        )

    def _candidate_moves(self, game_state: GameState) -> List[Move]:
        return candidate_moves(
            game_state,
            expansion_policy=self.expansion_policy,
            candidate_limit=self.candidate_limit,
        )


class MCTSAgent:
    """
    MCTS agent implementation.
    """

    VALID_ROLLOUT_POLICIES = {"random", "heuristic"}
    VALID_EXPANSION_POLICIES = {"uniform", "heuristic"}

    def __init__(
        self,
        num_rounds=2500,
        temperature=1.0,
        exploration_weight=0.6,
        max_rollout_depth=10,
        rollout_policy="heuristic",
        expansion_policy="heuristic",
        use_prior_bonus=True,
        candidate_limit=12,
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
        self.candidate_limit = candidate_limit

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

        forced_move = forced_center_opening_move(game_state)
        if forced_move is not None:
            return forced_move

        legal = game_state.legal_moves()
        playable = [m for m in legal if m.is_play]
        if not playable:
            return Move.pass_turn()

        root = MCTSNode(
            game_state,
            expansion_policy=self.expansion_policy,
            use_prior_bonus=self.use_prior_bonus,
            candidate_limit=self.candidate_limit,
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
        Run rollout with two retained optimizations:
        1. heuristic move selection
        2. rollout depth cap
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
        return select_rollout_move(game_state, self.rollout_policy)

    def _fast_position_eval(self, game_state: GameState, perspective_player: Player) -> float:
        return fast_position_value(game_state, perspective_player)

    def _effective_rounds(self, game_state: GameState) -> int:
        if self.num_rounds != -1:
            return max(1, int(self.num_rounds))
        board_area = game_state.board.num_rows * game_state.board.num_cols
        return max(320, board_area * 24)

    @staticmethod
    def _move_index(game_state: GameState) -> int:
        return move_index(game_state)
