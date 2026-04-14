"""
Reproducible experiment runner for the 5x5 Go homework.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.mcts_agent import MCTSAgent
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent
from dlgo import GameState, Player, compute_game_result
from dlgo.goboard import Move


MOVE_LIMIT_FACTOR = 2


@dataclass(frozen=True)
class AgentConfig:
    label: str
    family: str
    params: dict
    factory: Callable[[], object]


@dataclass(frozen=True)
class MatchupConfig:
    order: int
    group: str
    key: str
    display_name: str
    agent_a: AgentConfig
    agent_b: AgentConfig


def build_agent_catalog() -> dict[str, AgentConfig]:
    return {
        "random": AgentConfig(
            label="Random",
            family="Random",
            params={},
            factory=lambda: RandomAgent(),
        ),
        "mcts_standard_100": AgentConfig(
            label="MCTS standard",
            family="MCTS",
            params={
                "num_rounds": 100,
                "max_rollout_depth": -1,
                "rollout_policy": "random",
                "expansion_policy": "uniform",
                "use_prior_bonus": False,
            },
            factory=lambda: MCTSAgent.standard_baseline(num_rounds=100),
        ),
        "mcts_depth_cap_100": AgentConfig(
            label="MCTS + depth cap(20)",
            family="MCTS",
            params={
                "num_rounds": 100,
                "max_rollout_depth": 20,
                "rollout_policy": "random",
                "expansion_policy": "uniform",
                "use_prior_bonus": False,
            },
            factory=lambda: MCTSAgent.standard_baseline(
                num_rounds=100,
                max_rollout_depth=20,
            ),
        ),
        "mcts_heur_rollout_100": AgentConfig(
            label="MCTS + heuristic rollout",
            family="MCTS",
            params={
                "num_rounds": 100,
                "max_rollout_depth": 20,
                "rollout_policy": "heuristic",
                "expansion_policy": "uniform",
                "use_prior_bonus": False,
            },
            factory=lambda: MCTSAgent(
                num_rounds=100,
                max_rollout_depth=20,
                rollout_policy="heuristic",
                expansion_policy="uniform",
                use_prior_bonus=False,
            ),
        ),
        "mcts_full_opt_30": AgentConfig(
            label="MCTS optimized (30 rounds)",
            family="MCTS",
            params={
                "num_rounds": 30,
                "max_rollout_depth": 20,
                "rollout_policy": "heuristic",
                "expansion_policy": "heuristic",
                "use_prior_bonus": True,
            },
            factory=lambda: MCTSAgent(
                num_rounds=30,
                max_rollout_depth=20,
                rollout_policy="heuristic",
                expansion_policy="heuristic",
                use_prior_bonus=True,
            ),
        ),
        "mcts_full_opt_100": AgentConfig(
            label="MCTS optimized (100 rounds)",
            family="MCTS",
            params={
                "num_rounds": 100,
                "max_rollout_depth": 20,
                "rollout_policy": "heuristic",
                "expansion_policy": "heuristic",
                "use_prior_bonus": True,
            },
            factory=lambda: MCTSAgent(
                num_rounds=100,
                max_rollout_depth=20,
                rollout_policy="heuristic",
                expansion_policy="heuristic",
                use_prior_bonus=True,
            ),
        ),
        "mcts_full_opt_200": AgentConfig(
            label="MCTS optimized (200 rounds)",
            family="MCTS",
            params={
                "num_rounds": 200,
                "max_rollout_depth": 20,
                "rollout_policy": "heuristic",
                "expansion_policy": "heuristic",
                "use_prior_bonus": True,
            },
            factory=lambda: MCTSAgent(
                num_rounds=200,
                max_rollout_depth=20,
                rollout_policy="heuristic",
                expansion_policy="heuristic",
                use_prior_bonus=True,
            ),
        ),
        "minimax_depth_2": AgentConfig(
            label="Minimax depth=2",
            family="Minimax",
            params={"max_depth": 2, "max_branch": 12},
            factory=lambda: MinimaxAgent(max_depth=2, max_branch=12),
        ),
        "minimax_depth_3": AgentConfig(
            label="Minimax depth=3",
            family="Minimax",
            params={"max_depth": 3, "max_branch": 12},
            factory=lambda: MinimaxAgent(max_depth=3, max_branch=12),
        ),
    }


def build_matchups() -> list[MatchupConfig]:
    catalog = build_agent_catalog()
    random_agent = catalog["random"]

    return [
        MatchupConfig(
            order=0,
            group="mcts_ablation",
            key="mcts_standard_vs_random",
            display_name="Standard MCTS vs Random",
            agent_a=catalog["mcts_standard_100"],
            agent_b=random_agent,
        ),
        MatchupConfig(
            order=1,
            group="mcts_ablation",
            key="mcts_depth_cap_vs_random",
            display_name="MCTS + depth cap vs Random",
            agent_a=catalog["mcts_depth_cap_100"],
            agent_b=random_agent,
        ),
        MatchupConfig(
            order=2,
            group="mcts_ablation",
            key="mcts_heur_rollout_vs_random",
            display_name="MCTS + heuristic rollout vs Random",
            agent_a=catalog["mcts_heur_rollout_100"],
            agent_b=random_agent,
        ),
        MatchupConfig(
            order=3,
            group="mcts_ablation",
            key="mcts_full_opt_vs_random",
            display_name="MCTS optimized vs Random",
            agent_a=catalog["mcts_full_opt_100"],
            agent_b=random_agent,
        ),
        MatchupConfig(
            order=4,
            group="mcts_rounds",
            key="mcts_opt_30_vs_random",
            display_name="Optimized MCTS (30) vs Random",
            agent_a=catalog["mcts_full_opt_30"],
            agent_b=random_agent,
        ),
        MatchupConfig(
            order=5,
            group="mcts_rounds",
            key="mcts_opt_100_vs_random",
            display_name="Optimized MCTS (100) vs Random",
            agent_a=catalog["mcts_full_opt_100"],
            agent_b=random_agent,
        ),
        MatchupConfig(
            order=6,
            group="mcts_rounds",
            key="mcts_opt_200_vs_random",
            display_name="Optimized MCTS (200) vs Random",
            agent_a=catalog["mcts_full_opt_200"],
            agent_b=random_agent,
        ),
        MatchupConfig(
            order=7,
            group="minimax_depth",
            key="minimax_2_vs_random",
            display_name="Minimax depth=2 vs Random",
            agent_a=catalog["minimax_depth_2"],
            agent_b=random_agent,
        ),
        MatchupConfig(
            order=8,
            group="minimax_depth",
            key="minimax_3_vs_random",
            display_name="Minimax depth=3 vs Random",
            agent_a=catalog["minimax_depth_3"],
            agent_b=random_agent,
        ),
        MatchupConfig(
            order=9,
            group="crossplay",
            key="mcts_opt_30_vs_minimax_3",
            display_name="Optimized MCTS (30) vs Minimax depth=3",
            agent_a=catalog["mcts_full_opt_30"],
            agent_b=catalog["minimax_depth_3"],
        ),
        MatchupConfig(
            order=10,
            group="crossplay",
            key="mcts_opt_100_vs_minimax_3",
            display_name="Optimized MCTS (100) vs Minimax depth=3",
            agent_a=catalog["mcts_full_opt_100"],
            agent_b=catalog["minimax_depth_3"],
        ),
        MatchupConfig(
            order=11,
            group="crossplay",
            key="mcts_opt_200_vs_minimax_3",
            display_name="Optimized MCTS (200) vs Minimax depth=3",
            agent_a=catalog["mcts_full_opt_200"],
            agent_b=catalog["minimax_depth_3"],
        ),
        MatchupConfig(
            order=12,
            group="crossplay",
            key="mcts_opt_100_vs_minimax_2",
            display_name="Optimized MCTS (100) vs Minimax depth=2",
            agent_a=catalog["mcts_full_opt_100"],
            agent_b=catalog["minimax_depth_2"],
        ),
    ]


def player_name(player: Player | None) -> str:
    if player is None:
        return "none"
    return player.name


def signed_black_margin(game_result) -> float:
    return game_result.b - (game_result.w + game_result.komi)


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def format_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def to_float(value) -> float:
    return float(value)


def play_single_game(
    matchup: MatchupConfig,
    size: int,
    move_limit: int,
    seed: int,
    agent_a_as_black: bool,
    game_index: int,
) -> dict:
    random.seed(seed)

    black_cfg = matchup.agent_a if agent_a_as_black else matchup.agent_b
    white_cfg = matchup.agent_b if agent_a_as_black else matchup.agent_a
    black_agent = black_cfg.factory()
    white_agent = white_cfg.factory()

    agents = {
        Player.black: black_agent,
        Player.white: white_agent,
    }
    labels = {
        Player.black: black_cfg.label,
        Player.white: white_cfg.label,
    }

    game = GameState.new_game(size)
    move_count = 0
    turn_counts = {Player.black: 0, Player.white: 0}
    think_times = {Player.black: 0.0, Player.white: 0.0}
    start_time = time.perf_counter()

    while not game.is_over() and move_count < move_limit:
        current_player = game.next_player
        turn_start = time.perf_counter()
        move = agents[current_player].select_move(game)
        think_times[current_player] += time.perf_counter() - turn_start
        turn_counts[current_player] += 1

        if move is None or not game.is_valid_move(move):
            move = Move.pass_turn()

        game = game.apply_move(move)
        move_count += 1

    total_duration = time.perf_counter() - start_time

    if game.is_over():
        termination_reason = "resign" if game.last_move.is_resign else "two_passes"
        winner = game.winner()
        game_result = compute_game_result(game)
    else:
        termination_reason = "move_limit_score"
        winner = None
        game_result = compute_game_result(game)
        winner = game_result.winner

    black_margin = signed_black_margin(game_result)
    agent_a_color = Player.black if agent_a_as_black else Player.white
    agent_b_color = agent_a_color.other
    agent_a_margin = black_margin if agent_a_color == Player.black else -black_margin

    return {
        "group": matchup.group,
        "matchup_order": matchup.order,
        "config_key": matchup.key,
        "display_name": matchup.display_name,
        "seed": seed,
        "game_index": game_index,
        "agent_a_label": matchup.agent_a.label,
        "agent_b_label": matchup.agent_b.label,
        "agent_a_family": matchup.agent_a.family,
        "agent_b_family": matchup.agent_b.family,
        "agent_a_color": player_name(agent_a_color),
        "agent_b_color": player_name(agent_b_color),
        "black_label": labels[Player.black],
        "white_label": labels[Player.white],
        "winner_color": player_name(winner),
        "winner_label": labels.get(winner, "none"),
        "agent_a_won": int(winner == agent_a_color),
        "agent_b_won": int(winner == agent_b_color),
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
        "black_margin": black_margin,
        "agent_a_margin": agent_a_margin,
        "score_b": game_result.b,
        "score_w": game_result.w,
        "komi": game_result.komi,
        "result_text": str(game_result),
    }


def summarize_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["group"], row["config_key"]), []).append(row)

    summary_rows = []
    for group, config_key in sorted(grouped.keys()):
        group_rows = grouped[(group, config_key)]
        terminations = Counter(row["termination_reason"] for row in group_rows)
        display_name = group_rows[0]["display_name"]
        agent_a_label = group_rows[0]["agent_a_label"]
        agent_b_label = group_rows[0]["agent_b_label"]
        matchup_order = group_rows[0]["matchup_order"]

        summary_rows.append(
            {
                "group": group,
                "matchup_order": matchup_order,
                "config_key": config_key,
                "display_name": display_name,
                "agent_a_label": agent_a_label,
                "agent_b_label": agent_b_label,
                "games": len(group_rows),
                "win_rate": sum(row["agent_a_won"] for row in group_rows)
                / len(group_rows),
                "win_rate_percent": 100.0
                * sum(row["agent_a_won"] for row in group_rows)
                / len(group_rows),
                "avg_total_duration_s": sum(row["total_duration_s"] for row in group_rows)
                / len(group_rows),
                "avg_move_count": sum(row["move_count"] for row in group_rows)
                / len(group_rows),
                "avg_agent_a_move_time_s": sum(
                    (
                        row["black_avg_move_time_s"]
                        if row["agent_a_color"] == "black"
                        else row["white_avg_move_time_s"]
                    )
                    for row in group_rows
                )
                / len(group_rows),
                "avg_agent_b_move_time_s": sum(
                    (
                        row["white_avg_move_time_s"]
                        if row["agent_a_color"] == "black"
                        else row["black_avg_move_time_s"]
                    )
                    for row in group_rows
                )
                / len(group_rows),
                "avg_margin_for_agent_a": sum(
                    row["agent_a_margin"] for row in group_rows
                )
                / len(group_rows),
                "avg_abs_margin": sum(abs(row["agent_a_margin"]) for row in group_rows)
                / len(group_rows),
                "termination_summary": "; ".join(
                    f"{key}:{terminations[key]}"
                    for key in sorted(terminations.keys())
                ),
            }
        )

    summary_rows.sort(key=lambda row: row["matchup_order"])
    return summary_rows


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


def rows_for_group(summary_rows: list[dict], group: str) -> list[dict]:
    return [row for row in summary_rows if row["group"] == group]


def write_group_plot_csv(
    path: Path,
    rows: list[dict],
    x_key: str,
    time_key: str = "avg_total_duration_s",
) -> None:
    fieldnames = [x_key, "win_rate_percent", time_key]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    x_key: row[x_key],
                    "win_rate_percent": format_float(to_float(row["win_rate_percent"])),
                    time_key: format_float(
                        to_float(row[time_key]), 3
                    ),
                }
            )


def latex_table(rows: list[dict], title_mode: str) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrl}",
        r"\toprule",
    ]

    if title_mode == "crossplay":
        header = (
            r"Matchup & Win rate (\%) & Avg. time (s) & Avg. think (s) & "
            r"Avg. moves & Avg. margin & End reason \\"
        )
        lines.append(header)
        lines.append(r"\midrule")
        for row in rows:
            lines.append(
                " & ".join(
                    [
                        escape_latex(row["display_name"]),
                        format_float(row["win_rate_percent"]),
                        format_float(row["avg_total_duration_s"]),
                        format_float(row["avg_agent_a_move_time_s"]),
                        format_float(row["avg_move_count"]),
                        format_float(row["avg_margin_for_agent_a"]),
                        escape_latex(row["termination_summary"]),
                    ]
                )
                + r" \\"
            )
    else:
        header = (
            r"Config & Win rate (\%) & Avg. time (s) & Avg. think (s) & "
            r"Avg. moves & Avg. margin & End reason \\"
        )
        lines.append(header)
        lines.append(r"\midrule")
        for row in rows:
            lines.append(
                " & ".join(
                    [
                        escape_latex(row["agent_a_label"]),
                        format_float(row["win_rate_percent"]),
                        format_float(row["avg_total_duration_s"]),
                        format_float(row["avg_agent_a_move_time_s"]),
                        format_float(row["avg_move_count"]),
                        format_float(row["avg_margin_for_agent_a"]),
                        escape_latex(row["termination_summary"]),
                    ]
                )
                + r" \\"
            )

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def write_report_fragments(
    report_generated_dir: Path,
    summary_rows: list[dict],
    games_per_side: int,
    board_size: int,
) -> None:
    report_generated_dir.mkdir(parents=True, exist_ok=True)

    grouped = {
        "mcts_ablation": rows_for_group(summary_rows, "mcts_ablation"),
        "mcts_rounds": rows_for_group(summary_rows, "mcts_rounds"),
        "minimax_depth": rows_for_group(summary_rows, "minimax_depth"),
        "crossplay": rows_for_group(summary_rows, "crossplay"),
    }

    for name, rows in grouped.items():
        mode = "crossplay" if name == "crossplay" else "default"
        (report_generated_dir / f"{name}_table.tex").write_text(
            latex_table(rows, mode),
            encoding="utf-8",
        )

    metadata = "\n".join(
        [
            rf"\newcommand{{\ExperimentBoardSize}}{{{board_size}}}",
            rf"\newcommand{{\ExperimentGamesPerSide}}{{{games_per_side}}}",
            rf"\newcommand{{\ExperimentGamesPerConfig}}{{{games_per_side * 2}}}",
            "",
        ]
    )
    (report_generated_dir / "experiment_metadata.tex").write_text(
        metadata,
        encoding="utf-8",
    )

    ablation_plot_rows = []
    for row in grouped["mcts_ablation"]:
        ablation_plot_rows.append(
            {
                "label": row["agent_a_label"],
                "win_rate_percent": row["win_rate_percent"],
                "avg_total_duration_s": row["avg_total_duration_s"],
            }
        )
    write_group_plot_csv(
        report_generated_dir / "mcts_ablation_plot.csv",
        ablation_plot_rows,
        "label",
    )

    rounds_plot_rows = []
    round_order = {"30": 30, "100": 100, "200": 200}
    for row in grouped["mcts_rounds"]:
        rounds_value = None
        for token in round_order:
            if token in row["agent_a_label"]:
                rounds_value = round_order[token]
                break
        rounds_plot_rows.append(
            {
                "rounds": rounds_value,
                "win_rate_percent": row["win_rate_percent"],
                "avg_total_duration_s": row["avg_total_duration_s"],
            }
        )
    rounds_plot_rows.sort(key=lambda item: item["rounds"])
    write_group_plot_csv(
        report_generated_dir / "mcts_rounds_plot.csv",
        rounds_plot_rows,
        "rounds",
    )

    minimax_plot_rows = []
    for row in grouped["minimax_depth"]:
        depth = 2 if "depth=2" in row["agent_a_label"] else 3
        minimax_plot_rows.append(
            {
                "depth": depth,
                "win_rate_percent": row["win_rate_percent"],
                "avg_total_duration_s": row["avg_total_duration_s"],
            }
        )
    minimax_plot_rows.sort(key=lambda item: item["depth"])
    write_group_plot_csv(
        report_generated_dir / "minimax_depth_plot.csv",
        minimax_plot_rows,
        "depth",
    )

    crossplay_rows = grouped["crossplay"]
    if crossplay_rows:
        crossplay_compare_rows = []
        highlighted = next(
            (
                row
                for row in crossplay_rows
                if row["config_key"] == "mcts_opt_100_vs_minimax_3"
            ),
            crossplay_rows[0],
        )
        crossplay_compare_rows.extend(
            [
                {
                    "agent": highlighted["agent_a_label"],
                    "win_rate_percent": to_float(highlighted["win_rate_percent"]),
                    "avg_move_time_s": to_float(highlighted["avg_agent_a_move_time_s"]),
                },
                {
                    "agent": highlighted["agent_b_label"],
                    "win_rate_percent": 100.0 - to_float(highlighted["win_rate_percent"]),
                    "avg_move_time_s": to_float(highlighted["avg_agent_b_move_time_s"]),
                },
            ]
        )
        write_group_plot_csv(
            report_generated_dir / "crossplay_compare_plot.csv",
            crossplay_compare_rows,
            "agent",
            time_key="avg_move_time_s",
        )

        crossplay_round_rows = []
        for row in crossplay_rows:
            if row["agent_b_label"] != "Minimax depth=3":
                continue
            rounds = None
            for token in ("30", "100", "200"):
                if f"({token} rounds)" in row["agent_a_label"]:
                    rounds = int(token)
                    break
            if rounds is None:
                continue
            crossplay_round_rows.append(
                {
                    "rounds": rounds,
                    "win_rate_percent": to_float(row["win_rate_percent"]),
                    "avg_total_duration_s": to_float(row["avg_total_duration_s"]),
                }
            )
        crossplay_round_rows.sort(key=lambda item: item["rounds"])
        if crossplay_round_rows:
            write_group_plot_csv(
                report_generated_dir / "crossplay_mcts_rounds_plot.csv",
                crossplay_round_rows,
                "rounds",
            )

        crossplay_depth_rows = []
        for row in crossplay_rows:
            if row["agent_a_label"] != "MCTS optimized (100 rounds)":
                continue
            depth = None
            if row["agent_b_label"] == "Minimax depth=2":
                depth = 2
            elif row["agent_b_label"] == "Minimax depth=3":
                depth = 3
            if depth is None:
                continue
            crossplay_depth_rows.append(
                {
                    "depth": depth,
                    "win_rate_percent": to_float(row["win_rate_percent"]),
                    "avg_total_duration_s": to_float(row["avg_total_duration_s"]),
                }
            )
        crossplay_depth_rows.sort(key=lambda item: item["depth"])
        if crossplay_depth_rows:
            write_group_plot_csv(
                report_generated_dir / "crossplay_minimax_depth_plot.csv",
                crossplay_depth_rows,
                "depth",
            )


def build_selected_matchups(group: str) -> list[MatchupConfig]:
    matchups = build_matchups()
    if group == "all":
        return matchups
    return [matchup for matchup in matchups if matchup.group == group]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Go homework experiments.")
    parser.add_argument(
        "--group",
        choices=["all", "mcts_ablation", "mcts_rounds", "minimax_depth", "crossplay"],
        default="all",
        help="Experiment group to run.",
    )
    parser.add_argument(
        "--games-per-side",
        type=int,
        default=3,
        help="Number of seeds / games for each color assignment.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=5,
        help="Board size. Homework results should stay on 5x5.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments"),
        help="Directory for CSV / JSON outputs.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="First random seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_matchups = build_selected_matchups(args.group)
    move_limit = args.size * args.size * MOVE_LIMIT_FACTOR
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else REPO_ROOT / args.output_dir
    )

    per_game_rows = []
    for matchup in selected_matchups:
        for seed_offset in range(args.games_per_side):
            seed = args.seed_base + seed_offset
            per_game_rows.append(
                play_single_game(
                    matchup,
                    size=args.size,
                    move_limit=move_limit,
                    seed=seed,
                    agent_a_as_black=True,
                    game_index=len(per_game_rows),
                )
            )
            per_game_rows.append(
                play_single_game(
                    matchup,
                    size=args.size,
                    move_limit=move_limit,
                    seed=seed,
                    agent_a_as_black=False,
                    game_index=len(per_game_rows),
                )
            )

    summary_rows = summarize_rows(per_game_rows)

    per_game_fieldnames = [
        "group",
        "matchup_order",
        "config_key",
        "display_name",
        "seed",
        "game_index",
        "agent_a_label",
        "agent_b_label",
        "agent_a_family",
        "agent_b_family",
        "agent_a_color",
        "agent_b_color",
        "black_label",
        "white_label",
        "winner_color",
        "winner_label",
        "agent_a_won",
        "agent_b_won",
        "move_count",
        "termination_reason",
        "total_duration_s",
        "black_turns",
        "white_turns",
        "black_think_time_s",
        "white_think_time_s",
        "black_avg_move_time_s",
        "white_avg_move_time_s",
        "black_margin",
        "agent_a_margin",
        "score_b",
        "score_w",
        "komi",
        "result_text",
    ]
    summary_fieldnames = [
        "group",
        "matchup_order",
        "config_key",
        "display_name",
        "agent_a_label",
        "agent_b_label",
        "games",
        "win_rate",
        "win_rate_percent",
        "avg_total_duration_s",
        "avg_move_count",
        "avg_agent_a_move_time_s",
        "avg_agent_b_move_time_s",
        "avg_margin_for_agent_a",
        "avg_abs_margin",
        "termination_summary",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_game.csv", per_game_rows, per_game_fieldnames)
    write_csv(output_dir / "summary.csv", summary_rows, summary_fieldnames)
    write_json(
        output_dir / "summary.json",
        {
            "metadata": {
                "group": args.group,
                "board_size": args.size,
                "games_per_side": args.games_per_side,
                "games_per_config": args.games_per_side * 2,
                "seed_base": args.seed_base,
                "move_limit": move_limit,
            },
            "summary": summary_rows,
        },
    )

    write_report_fragments(
        REPO_ROOT / "report" / "generated",
        summary_rows,
        games_per_side=args.games_per_side,
        board_size=args.size,
    )

    print(f"Saved {len(per_game_rows)} game records to {output_dir / 'per_game.csv'}")
    print(f"Saved {len(summary_rows)} summary rows to {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
