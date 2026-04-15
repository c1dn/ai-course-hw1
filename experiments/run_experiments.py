"""
Flexible experiment runner for Go agents.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.mcts_agent import MCTSAgent
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent
from dlgo import GameState, Player, compute_game_result, default_komi_for_board_size
from dlgo.gotypes import Point
from dlgo.goboard import Move


MOVE_LIMIT_FACTOR = 2
DEFAULT_MOVE_TIME_LIMIT_S = 20.0
DEFAULT_RESULTS_ROOT = REPO_ROOT / "experiments" / "results"


@dataclass(frozen=True)
class AgentSpec:
    family: str
    label: str
    params: dict
    factory: Callable[[], object]


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def player_name(player: Player | None) -> str:
    if player is None:
        return "none"
    return player.name


def signed_black_margin(game_result) -> float:
    return game_result.b - (game_result.w + game_result.komi)


def move_to_text(move: Move) -> str:
    if move.is_pass:
        return "pass"
    if move.is_resign:
        return "resign"
    return f"({move.point.row}, {move.point.col})"


def render_board_unicode(game_state: GameState) -> str:
    board = game_state.board
    header = "   " + " ".join(f"{col:2}" for col in range(1, board.num_cols + 1))
    lines = [header]
    for row in range(1, board.num_rows + 1):
        cells = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(Point(row, col))
            if stone == Player.black:
                cells.append("●")
            elif stone == Player.white:
                cells.append("○")
            else:
                cells.append("·")
        lines.append(f"{row:2} " + " ".join(f"{cell:>2}" for cell in cells))
    return "\n".join(lines)


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "experiment"


def parse_bool_arg(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value, got {value!r}. Use true/false."
    )


def build_agent_spec(side: str, args: argparse.Namespace) -> AgentSpec:
    agent_name = getattr(args, f"{side}_agent")

    if agent_name == "random":
        return AgentSpec(
            family="Random",
            label="Random",
            params={},
            factory=lambda: RandomAgent(),
        )

    if agent_name == "mcts":
        rounds = getattr(args, f"{side}_mcts_rounds")
        exploration_weight = getattr(args, f"{side}_mcts_exploration_weight")
        max_rollout_depth = getattr(args, f"{side}_mcts_max_rollout_depth")
        rollout_policy = getattr(args, f"{side}_mcts_rollout_policy")
        expansion_policy = getattr(args, f"{side}_mcts_expansion_policy")
        use_prior_bonus = getattr(args, f"{side}_mcts_use_prior_bonus")
        if rounds < 1:
            raise ValueError(f"{side} MCTS rounds must be >= 1, got {rounds}.")
        params = {
            "num_rounds": rounds,
            "exploration_weight": exploration_weight,
            "max_rollout_depth": max_rollout_depth,
            "rollout_policy": rollout_policy,
            "expansion_policy": expansion_policy,
            "use_prior_bonus": use_prior_bonus,
        }
        return AgentSpec(
            family="MCTS",
            label=(
                f"MCTS(rounds={rounds}, c={exploration_weight:g}, "
                f"depth={max_rollout_depth}, rollout={rollout_policy}, "
                f"expand={expansion_policy}, prior={str(use_prior_bonus).lower()})"
            ),
            params=params,
            factory=lambda: MCTSAgent(
                num_rounds=rounds,
                exploration_weight=exploration_weight,
                max_rollout_depth=max_rollout_depth,
                rollout_policy=rollout_policy,
                expansion_policy=expansion_policy,
                use_prior_bonus=use_prior_bonus,
            ),
        )

    if agent_name == "minimax":
        depth = getattr(args, f"{side}_minimax_depth")
        if depth < 1:
            raise ValueError(f"{side} minimax depth must be >= 1, got {depth}.")
        params = {"max_depth": depth}
        return AgentSpec(
            family="Minimax",
            label=f"Minimax(depth={depth})",
            params=params,
            factory=lambda: MinimaxAgent(max_depth=depth),
        )

    raise ValueError(f"Unsupported {side} agent: {agent_name!r}")


def resolve_output_dir(
    args: argparse.Namespace,
    black_spec: AgentSpec,
    white_spec: AgentSpec,
) -> Path:
    if args.output_dir is not None:
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir
        return output_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{slugify(black_spec.label)}_vs_{slugify(white_spec.label)}"
        f"_size{args.size}_games{args.games}_{timestamp}"
    )
    return DEFAULT_RESULTS_ROOT / run_name


def play_single_game(
    size: int,
    move_limit: int,
    komi: float | None,
    seed: int,
    game_index: int,
    black_spec: AgentSpec,
    white_spec: AgentSpec,
    move_time_limit_s: float,
    trace_moves: bool,
) -> dict:
    random.seed(seed)

    black_agent = black_spec.factory()
    white_agent = white_spec.factory()
    agents = {
        Player.black: black_agent,
        Player.white: white_agent,
    }
    labels = {
        Player.black: black_spec.label,
        Player.white: white_spec.label,
    }

    game = GameState.new_game(size, komi=komi)
    move_count = 0
    turn_counts = {Player.black: 0, Player.white: 0}
    think_times = {Player.black: 0.0, Player.white: 0.0}
    max_move_times = {Player.black: 0.0, Player.white: 0.0}
    start_time = time.perf_counter()

    while not game.is_over() and move_count < move_limit:
        current_player = game.next_player
        turn_start = time.perf_counter()
        move = agents[current_player].select_move(game)
        elapsed = time.perf_counter() - turn_start

        think_times[current_player] += elapsed
        max_move_times[current_player] = max(max_move_times[current_player], elapsed)
        turn_counts[current_player] += 1

        if move is None or not game.is_valid_move(move):
            move = Move.pass_turn()

        game = game.apply_move(move)
        move_count += 1

        if trace_moves:
            print(
                "[trace] "
                f"game={game_index} | seed={seed} | move={move_count} | "
                f"player={current_player.name} | agent={labels[current_player]} | "
                f"action={move_to_text(move)} | elapsed={elapsed:.3f}s"
            )
            print(render_board_unicode(game))
            print()

    total_duration = time.perf_counter() - start_time

    if game.is_over():
        termination_reason = "resign" if game.last_move.is_resign else "two_passes"
        winner = game.winner()
        game_result = compute_game_result(game)
    else:
        termination_reason = "move_limit_score"
        game_result = compute_game_result(game)
        winner = game_result.winner

    black_margin = signed_black_margin(game_result)

    return {
        "game_index": game_index,
        "seed": seed,
        "board_size": size,
        "move_limit": move_limit,
        "komi": game_result.komi,
        "black_label": black_spec.label,
        "white_label": white_spec.label,
        "black_family": black_spec.family,
        "white_family": white_spec.family,
        "winner_color": player_name(winner),
        "winner_label": labels.get(winner, "none"),
        "black_won": int(winner == Player.black),
        "white_won": int(winner == Player.white),
        "move_count": move_count,
        "termination_reason": termination_reason,
        "total_duration_s": total_duration,
        "black_turns": turn_counts[Player.black],
        "white_turns": turn_counts[Player.white],
        "black_think_time_s": think_times[Player.black],
        "white_think_time_s": think_times[Player.white],
        "black_avg_move_time_s": think_times[Player.black]
        / max(1, turn_counts[Player.black]),
        "white_avg_move_time_s": think_times[Player.white]
        / max(1, turn_counts[Player.white]),
        "black_max_move_time_s": max_move_times[Player.black],
        "white_max_move_time_s": max_move_times[Player.white],
        "black_exceeded_move_time_limit": int(
            max_move_times[Player.black] > move_time_limit_s
        ),
        "white_exceeded_move_time_limit": int(
            max_move_times[Player.white] > move_time_limit_s
        ),
        "black_margin": black_margin,
        "score_b": game_result.b,
        "score_w": game_result.w,
        "komi": game_result.komi,
        "result_text": str(game_result),
    }


def summarize_rows(
    rows: list[dict],
    black_spec: AgentSpec,
    white_spec: AgentSpec,
    move_time_limit_s: float,
) -> dict:
    if not rows:
        raise ValueError("No game rows to summarize.")

    terminations = Counter(row["termination_reason"] for row in rows)

    return {
        "black_label": black_spec.label,
        "white_label": white_spec.label,
        "black_family": black_spec.family,
        "white_family": white_spec.family,
        "games": len(rows),
        "black_win_rate": sum(row["black_won"] for row in rows) / len(rows),
        "white_win_rate": sum(row["white_won"] for row in rows) / len(rows),
        "black_win_rate_percent": 100.0
        * sum(row["black_won"] for row in rows)
        / len(rows),
        "white_win_rate_percent": 100.0
        * sum(row["white_won"] for row in rows)
        / len(rows),
        "avg_total_duration_s": sum(row["total_duration_s"] for row in rows) / len(rows),
        "avg_move_count": sum(row["move_count"] for row in rows) / len(rows),
        "avg_black_move_time_s": sum(row["black_avg_move_time_s"] for row in rows)
        / len(rows),
        "avg_white_move_time_s": sum(row["white_avg_move_time_s"] for row in rows)
        / len(rows),
        "max_black_move_time_s": max(row["black_max_move_time_s"] for row in rows),
        "max_white_move_time_s": max(row["white_max_move_time_s"] for row in rows),
        "black_exceeded_move_time_limit_games": sum(
            row["black_exceeded_move_time_limit"] for row in rows
        ),
        "white_exceeded_move_time_limit_games": sum(
            row["white_exceeded_move_time_limit"] for row in rows
        ),
        "move_time_limit_s": move_time_limit_s,
        "avg_black_margin": sum(row["black_margin"] for row in rows) / len(rows),
        "avg_abs_margin": sum(abs(row["black_margin"]) for row in rows) / len(rows),
        "termination_summary": "; ".join(
            f"{key}:{terminations[key]}" for key in sorted(terminations.keys())
        ),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_seed_rows(path: Path, seeds: list[int]) -> None:
    """
    Write the explicit per-game seed list for reproducibility.
    """
    rows = [{"game_index": index, "seed": seed} for index, seed in enumerate(seeds)]
    write_csv(path, rows, ["game_index", "seed"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parameterized Go experiments.")
    parser.add_argument(
        "--size",
        type=int,
        default=5,
        help="Board size.",
    )
    parser.add_argument(
        "--black-agent",
        choices=["random", "mcts", "minimax"],
        required=True,
        help="Black-side agent.",
    )
    parser.add_argument(
        "--white-agent",
        choices=["random", "mcts", "minimax"],
        required=True,
        help="White-side agent.",
    )
    parser.add_argument(
        "--mcts-rounds",
        type=int,
        default=None,
        help=(
            "Convenience shortcut when exactly one side uses MCTS. "
            "Use side-specific flags only when both sides are MCTS."
        ),
    )
    parser.add_argument(
        "--mcts-exploration-weight",
        type=float,
        default=None,
        help=(
            "Convenience shortcut for the MCTS exploration constant when exactly one "
            "side uses MCTS. Use side-specific flags when both sides are MCTS."
        ),
    )
    parser.add_argument(
        "--mcts-max-rollout-depth",
        type=int,
        default=None,
        help=(
            "Convenience shortcut for MCTS rollout depth when exactly one side uses "
            "MCTS. Use side-specific flags when both sides are MCTS."
        ),
    )
    parser.add_argument(
        "--mcts-rollout-policy",
        choices=sorted(MCTSAgent.VALID_ROLLOUT_POLICIES),
        default=None,
        help=(
            "Convenience shortcut for MCTS rollout policy when exactly one side uses "
            "MCTS. Use side-specific flags when both sides are MCTS."
        ),
    )
    parser.add_argument(
        "--mcts-expansion-policy",
        choices=sorted(MCTSAgent.VALID_EXPANSION_POLICIES),
        default=None,
        help=(
            "Convenience shortcut for MCTS expansion policy when exactly one side uses "
            "MCTS. Use side-specific flags when both sides are MCTS."
        ),
    )
    parser.add_argument(
        "--mcts-use-prior-bonus",
        type=parse_bool_arg,
        default=None,
        metavar="{true,false}",
        help=(
            "Convenience shortcut for enabling/disabling MCTS prior bonus when exactly "
            "one side uses MCTS. Use side-specific flags when both sides are MCTS."
        ),
    )
    parser.add_argument(
        "--minimax-depth",
        type=int,
        default=None,
        help=(
            "Convenience shortcut when exactly one side uses Minimax. "
            "Use side-specific flags only when both sides are Minimax."
        ),
    )
    parser.add_argument(
        "--black-mcts-rounds",
        type=int,
        default=None,
        help="Black-side MCTS rounds. Only needed when both sides are MCTS.",
    )
    parser.add_argument(
        "--white-mcts-rounds",
        type=int,
        default=None,
        help="White-side MCTS rounds. Only needed when both sides are MCTS.",
    )
    parser.add_argument(
        "--black-mcts-exploration-weight",
        type=float,
        default=None,
        help="Black-side MCTS exploration constant.",
    )
    parser.add_argument(
        "--white-mcts-exploration-weight",
        type=float,
        default=None,
        help="White-side MCTS exploration constant.",
    )
    parser.add_argument(
        "--black-mcts-max-rollout-depth",
        type=int,
        default=None,
        help="Black-side MCTS rollout depth. Use -1 for rollout-to-terminal.",
    )
    parser.add_argument(
        "--white-mcts-max-rollout-depth",
        type=int,
        default=None,
        help="White-side MCTS rollout depth. Use -1 for rollout-to-terminal.",
    )
    parser.add_argument(
        "--black-mcts-rollout-policy",
        choices=sorted(MCTSAgent.VALID_ROLLOUT_POLICIES),
        default=None,
        help="Black-side MCTS rollout policy.",
    )
    parser.add_argument(
        "--white-mcts-rollout-policy",
        choices=sorted(MCTSAgent.VALID_ROLLOUT_POLICIES),
        default=None,
        help="White-side MCTS rollout policy.",
    )
    parser.add_argument(
        "--black-mcts-expansion-policy",
        choices=sorted(MCTSAgent.VALID_EXPANSION_POLICIES),
        default=None,
        help="Black-side MCTS expansion policy.",
    )
    parser.add_argument(
        "--white-mcts-expansion-policy",
        choices=sorted(MCTSAgent.VALID_EXPANSION_POLICIES),
        default=None,
        help="White-side MCTS expansion policy.",
    )
    parser.add_argument(
        "--black-mcts-use-prior-bonus",
        type=parse_bool_arg,
        default=None,
        metavar="{true,false}",
        help="Whether black-side MCTS uses prior bonus.",
    )
    parser.add_argument(
        "--white-mcts-use-prior-bonus",
        type=parse_bool_arg,
        default=None,
        metavar="{true,false}",
        help="Whether white-side MCTS uses prior bonus.",
    )
    parser.add_argument(
        "--black-minimax-depth",
        type=int,
        default=None,
        help="Black-side Minimax depth. Only needed when both sides are Minimax.",
    )
    parser.add_argument(
        "--white-minimax-depth",
        type=int,
        default=None,
        help="White-side Minimax depth. Only needed when both sides are Minimax.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="Number of games to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default is a timestamped subdirectory under experiments/results.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="First random seed.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit random seed for each game. When provided, the number of "
            "seeds must match --games and overrides --seed-base."
        ),
    )
    parser.add_argument(
        "--move-limit",
        type=int,
        default=None,
        help="Maximum moves per game. Default is board_size * board_size * 2.",
    )
    parser.add_argument(
        "--komi",
        type=float,
        default=None,
        help="Optional komi override. Default is 7.5.",
    )
    parser.add_argument(
        "--move-time-limit-s",
        type=float,
        default=DEFAULT_MOVE_TIME_LIMIT_S,
        help="Threshold used when checking whether a single move exceeds the limit.",
    )
    parser.add_argument(
        "--trace-moves",
        action="store_true",
        help="Print per-move elapsed time and a Unicode board after each move.",
    )
    args = parser.parse_args()
    resolve_agent_parameter_shortcuts(args)
    return args


def resolve_agent_parameter_shortcuts(args: argparse.Namespace) -> None:
    resolve_family_parameter(
        args=args,
        family_name="mcts",
        generic_attr="mcts_rounds",
        side_attrs=("black_mcts_rounds", "white_mcts_rounds"),
        side_agents=("black_agent", "white_agent"),
        default_value=100,
        label="MCTS rounds",
    )
    resolve_family_parameter(
        args=args,
        family_name="mcts",
        generic_attr="mcts_exploration_weight",
        side_attrs=("black_mcts_exploration_weight", "white_mcts_exploration_weight"),
        side_agents=("black_agent", "white_agent"),
        default_value=0.6,
        label="MCTS exploration weight",
    )
    resolve_family_parameter(
        args=args,
        family_name="mcts",
        generic_attr="mcts_max_rollout_depth",
        side_attrs=("black_mcts_max_rollout_depth", "white_mcts_max_rollout_depth"),
        side_agents=("black_agent", "white_agent"),
        default_value=10,
        label="MCTS max rollout depth",
    )
    resolve_family_parameter(
        args=args,
        family_name="mcts",
        generic_attr="mcts_rollout_policy",
        side_attrs=("black_mcts_rollout_policy", "white_mcts_rollout_policy"),
        side_agents=("black_agent", "white_agent"),
        default_value="heuristic",
        label="MCTS rollout policy",
    )
    resolve_family_parameter(
        args=args,
        family_name="mcts",
        generic_attr="mcts_expansion_policy",
        side_attrs=("black_mcts_expansion_policy", "white_mcts_expansion_policy"),
        side_agents=("black_agent", "white_agent"),
        default_value="heuristic",
        label="MCTS expansion policy",
    )
    resolve_family_parameter(
        args=args,
        family_name="mcts",
        generic_attr="mcts_use_prior_bonus",
        side_attrs=("black_mcts_use_prior_bonus", "white_mcts_use_prior_bonus"),
        side_agents=("black_agent", "white_agent"),
        default_value=True,
        label="MCTS prior bonus toggle",
    )
    resolve_family_parameter(
        args=args,
        family_name="minimax",
        generic_attr="minimax_depth",
        side_attrs=("black_minimax_depth", "white_minimax_depth"),
        side_agents=("black_agent", "white_agent"),
        default_value=3,
        label="Minimax depth",
    )


def resolve_family_parameter(
    *,
    args: argparse.Namespace,
    family_name: str,
    generic_attr: str,
    side_attrs: tuple[str, str],
    side_agents: tuple[str, str],
    default_value,
    label: str,
) -> None:
    generic_value = getattr(args, generic_attr)
    matching_sides = [
        side_attr
        for side_attr, side_agent in zip(side_attrs, side_agents)
        if getattr(args, side_agent) == family_name
    ]

    if generic_value is not None:
        if not matching_sides:
            raise ValueError(
                f"--{generic_attr.replace('_', '-')} was provided, "
                f"but neither side is using {family_name}."
            )
        if len(matching_sides) > 1:
            raise ValueError(
                f"--{generic_attr.replace('_', '-')} is ambiguous when both sides use "
                f"{family_name}. Please use side-specific flags for {label}."
            )
        target_attr = matching_sides[0]
        explicit_value = getattr(args, target_attr)
        if explicit_value is not None and explicit_value != generic_value:
            raise ValueError(
                f"Conflicting {label}: "
                f"--{generic_attr.replace('_', '-')}={generic_value} and "
                f"--{target_attr.replace('_', '-')}={explicit_value}."
            )
        setattr(args, target_attr, generic_value)

    for side_attr, side_agent in zip(side_attrs, side_agents):
        if getattr(args, side_agent) != family_name:
            continue
        if getattr(args, side_attr) is None:
            setattr(args, side_attr, default_value)


def resolve_game_seeds(args: argparse.Namespace) -> list[int]:
    """
    Resolve the per-game seed list from CLI arguments.
    """
    if args.seeds is not None:
        if len(args.seeds) != args.games:
            raise ValueError(
                f"--seeds expects exactly --games values. "
                f"Got {len(args.seeds)} seeds for --games={args.games}."
            )
        return list(args.seeds)
    return [args.seed_base + game_index for game_index in range(args.games)]


def main() -> None:
    args = parse_args()
    black_spec = build_agent_spec("black", args)
    white_spec = build_agent_spec("white", args)

    if args.games < 1:
        raise ValueError(f"--games must be >= 1, got {args.games}.")

    move_limit = args.move_limit
    if move_limit is None:
        move_limit = args.size * args.size * MOVE_LIMIT_FACTOR
    effective_komi = (
        args.komi if args.komi is not None else default_komi_for_board_size(args.size)
    )
    game_seeds = resolve_game_seeds(args)

    output_dir = resolve_output_dir(args, black_spec, white_spec)

    per_game_rows = []
    for game_index in range(args.games):
        per_game_rows.append(
            play_single_game(
                size=args.size,
                move_limit=move_limit,
                komi=args.komi,
                seed=game_seeds[game_index],
                game_index=game_index,
                black_spec=black_spec,
                white_spec=white_spec,
                move_time_limit_s=args.move_time_limit_s,
                trace_moves=args.trace_moves,
            )
        )

    summary_row = summarize_rows(
        per_game_rows,
        black_spec=black_spec,
        white_spec=white_spec,
        move_time_limit_s=args.move_time_limit_s,
    )

    per_game_fieldnames = [
        "game_index",
        "seed",
        "board_size",
        "move_limit",
        "komi",
        "black_label",
        "white_label",
        "black_family",
        "white_family",
        "winner_color",
        "winner_label",
        "black_won",
        "white_won",
        "move_count",
        "termination_reason",
        "total_duration_s",
        "black_turns",
        "white_turns",
        "black_think_time_s",
        "white_think_time_s",
        "black_avg_move_time_s",
        "white_avg_move_time_s",
        "black_max_move_time_s",
        "white_max_move_time_s",
        "black_exceeded_move_time_limit",
        "white_exceeded_move_time_limit",
        "black_margin",
        "score_b",
        "score_w",
        "komi",
        "result_text",
    ]
    summary_fieldnames = [
        "black_label",
        "white_label",
        "black_family",
        "white_family",
        "games",
        "black_win_rate",
        "white_win_rate",
        "black_win_rate_percent",
        "white_win_rate_percent",
        "avg_total_duration_s",
        "avg_move_count",
        "avg_black_move_time_s",
        "avg_white_move_time_s",
        "max_black_move_time_s",
        "max_white_move_time_s",
        "black_exceeded_move_time_limit_games",
        "white_exceeded_move_time_limit_games",
        "move_time_limit_s",
        "avg_black_margin",
        "avg_abs_margin",
        "termination_summary",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_game.csv", per_game_rows, per_game_fieldnames)
    write_csv(output_dir / "summary.csv", [summary_row], summary_fieldnames)
    write_seed_rows(output_dir / "game_seeds.csv", game_seeds)
    write_json(
        output_dir / "summary.json",
        {
            "metadata": {
                "board_size": args.size,
                "games": args.games,
                "seed_base": args.seed_base,
                "seeds": game_seeds,
                "move_limit": move_limit,
                "komi": effective_komi,
                "move_time_limit_s": args.move_time_limit_s,
                "black_agent": {
                    "family": black_spec.family,
                    "label": black_spec.label,
                    "params": black_spec.params,
                },
                "white_agent": {
                    "family": white_spec.family,
                    "label": white_spec.label,
                    "params": white_spec.params,
                },
            },
            "summary": summary_row,
            "per_game_count": len(per_game_rows),
        },
    )

    print(f"Output directory: {output_dir}")
    print(f"Black: {black_spec.label}")
    print(f"White: {white_spec.label}")
    print(f"Games: {args.games}")
    print(f"Komi: {format_float(effective_komi, 1)}")
    print(f"Saved per-game results to {output_dir / 'per_game.csv'}")
    print(f"Saved summary to {output_dir / 'summary.csv'}")
    print(
        "Max move time: "
        f"black={format_float(summary_row['max_black_move_time_s'])}s, "
        f"white={format_float(summary_row['max_white_move_time_s'])}s"
    )


if __name__ == "__main__":
    main()
