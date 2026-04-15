"""
Multi-process launcher for the experiment plans listed in todo.md.

Supported plans:
- crossplay: existing MCTS-vs-Minimax comparison jobs
- ablation: fixed-1000-round MCTS ablations against standard MCTS

Each logical job is executed as two 5-game subruns so that the tracked
side swaps color across the 10-game bundle, then the launcher merges
the two subruns back into a single per-job summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENTS = REPO_ROOT / "experiments" / "run_experiments.py"
CROSSPLAY_OUTPUT_ROOT = REPO_ROOT / "experiments" / "results" / "experiment_plan"
ABLATION_OUTPUT_ROOT = REPO_ROOT / "experiments" / "results" / "mcts_ablation_plan"


@dataclass(frozen=True)
class CrossplayJobSpec:
    key: str
    mcts_rounds: int
    minimax_depth: int
    games: int
    note: str = ""


@dataclass(frozen=True)
class MCTSConfig:
    rounds: int
    exploration_weight: float
    max_rollout_depth: int
    rollout_policy: str
    expansion_policy: str
    use_prior_bonus: bool

    def to_cli_args(self, side: str) -> list[str]:
        return [
            f"--{side}-mcts-rounds",
            str(self.rounds),
            f"--{side}-mcts-exploration-weight",
            f"{self.exploration_weight:.12g}",
            f"--{side}-mcts-max-rollout-depth",
            str(self.max_rollout_depth),
            f"--{side}-mcts-rollout-policy",
            self.rollout_policy,
            f"--{side}-mcts-expansion-policy",
            self.expansion_policy,
            f"--{side}-mcts-use-prior-bonus",
            str(self.use_prior_bonus).lower(),
        ]

    def label(self) -> str:
        return (
            f"MCTS(rounds={self.rounds}, c={self.exploration_weight:g}, "
            f"depth={self.max_rollout_depth}, rollout={self.rollout_policy}, "
            f"expand={self.expansion_policy}, prior={str(self.use_prior_bonus).lower()})"
        )


@dataclass(frozen=True)
class AblationJobSpec:
    key: str
    variant_name: str
    variant_config: MCTSConfig
    games: int
    note: str = ""
    baseline_name: str = "Standard MCTS"
    baseline_config: MCTSConfig | None = None

    def resolved_baseline_config(self) -> MCTSConfig:
        if self.baseline_config is None:
            raise ValueError(f"Missing baseline_config for {self.key}.")
        return self.baseline_config


@dataclass(frozen=True)
class GroupSpec:
    name: str
    jobs: tuple[str, ...]
    note: str = ""


CROSSPLAY_JOBS = (
    CrossplayJobSpec(
        "mcts300_vs_minimax4", mcts_rounds=300, minimax_depth=4, games=10, note="~1.5s level"
    ),
    CrossplayJobSpec(
        "mcts600_vs_minimax5", mcts_rounds=600, minimax_depth=5, games=10, note="~3.5s level"
    ),
    CrossplayJobSpec(
        "mcts2200_vs_minimax6",
        mcts_rounds=2200,
        minimax_depth=6,
        games=10,
        note="~25s level",
    ),
    CrossplayJobSpec("mcts2200_vs_minimax4", mcts_rounds=2200, minimax_depth=4, games=10),
    CrossplayJobSpec("mcts2200_vs_minimax5", mcts_rounds=2200, minimax_depth=5, games=10),
    CrossplayJobSpec("mcts3000_vs_minimax5", mcts_rounds=3000, minimax_depth=5, games=10),
    CrossplayJobSpec("mcts4000_vs_minimax5", mcts_rounds=4000, minimax_depth=5, games=10),
    CrossplayJobSpec("mcts5000_vs_minimax5", mcts_rounds=5000, minimax_depth=5, games=10),
)


CROSSPLAY_GROUPS = (
    GroupSpec(
        name="group1_depth_scaling",
        jobs=(
            "mcts300_vs_minimax4",
            "mcts600_vs_minimax5",
            "mcts2200_vs_minimax6",
        ),
        note="3 planned runs, 10 games each",
    ),
    GroupSpec(
        name="group2_depth_sweep_at_2200",
        jobs=(
            "mcts2200_vs_minimax4",
            "mcts2200_vs_minimax5",
            "mcts2200_vs_minimax6",
        ),
        note="3 comparison entries, 10 games each; mcts2200_vs_minimax6 is reused from group1",
    ),
    GroupSpec(
        name="group3_round_sweep_at_depth5",
        jobs=(
            "mcts2200_vs_minimax5",
            "mcts3000_vs_minimax5",
            "mcts4000_vs_minimax5",
            "mcts5000_vs_minimax5",
        ),
        note="mcts2200_vs_minimax5 is reused from group2; all runs use 10 games each",
    ),
)


STANDARD_MCTS_CONFIG = MCTSConfig(
    rounds=1000,
    exploration_weight=1.414,
    max_rollout_depth=50,
    rollout_policy="random",
    expansion_policy="uniform",
    use_prior_bonus=False,
)


ABLATION_JOBS = (
    AblationJobSpec(
        key="standard_vs_standard_1000",
        variant_name="Standard MCTS",
        variant_config=STANDARD_MCTS_CONFIG,
        games=10,
        note="sanity check",
        baseline_config=STANDARD_MCTS_CONFIG,
    ),
    AblationJobSpec(
        key="low_c_vs_standard_1000",
        variant_name="Low-c MCTS",
        variant_config=replace(STANDARD_MCTS_CONFIG, exploration_weight=0.6),
        games=10,
        note="only lower UCT exploration constant",
        baseline_config=STANDARD_MCTS_CONFIG,
    ),
    AblationJobSpec(
        key="cutoff10_eval_vs_standard_1000",
        variant_name="Cutoff-10 MCTS",
        variant_config=replace(STANDARD_MCTS_CONFIG, max_rollout_depth=10),
        games=10,
        note="depth cap with current cutoff evaluator",
        baseline_config=STANDARD_MCTS_CONFIG,
    ),
    AblationJobSpec(
        key="heuristic_rollout_vs_standard_1000",
        variant_name="Heuristic-rollout MCTS",
        variant_config=replace(STANDARD_MCTS_CONFIG, rollout_policy="heuristic"),
        games=10,
        note="only heuristic rollout policy",
        baseline_config=STANDARD_MCTS_CONFIG,
    ),
    AblationJobSpec(
        key="heuristic_expansion_vs_standard_1000",
        variant_name="Heuristic-expansion MCTS",
        variant_config=replace(STANDARD_MCTS_CONFIG, expansion_policy="heuristic"),
        games=10,
        note="only heuristic expansion policy",
        baseline_config=STANDARD_MCTS_CONFIG,
    ),
    AblationJobSpec(
        key="prior_bonus_vs_standard_1000",
        variant_name="Prior-bonus MCTS",
        variant_config=replace(STANDARD_MCTS_CONFIG, use_prior_bonus=True),
        games=10,
        note="only prior bonus enabled",
        baseline_config=STANDARD_MCTS_CONFIG,
    ),
    AblationJobSpec(
        key="full_optimized_vs_standard_1000",
        variant_name="Optimized MCTS",
        variant_config=replace(
            STANDARD_MCTS_CONFIG,
            exploration_weight=0.6,
            max_rollout_depth=10,
            rollout_policy="heuristic",
            expansion_policy="heuristic",
            use_prior_bonus=True,
        ),
        games=10,
        note="current optimized configuration vs standard",
        baseline_config=STANDARD_MCTS_CONFIG,
    ),
)


ABLATION_GROUPS = (
    GroupSpec(
        name="group1_standard_mcts_ablation",
        jobs=tuple(job.key for job in ABLATION_JOBS),
        note=(
            "All jobs fix num_rounds=1000 and compare one MCTS ablation variant "
            "against standard MCTS over 10 games with swapped colors."
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the todo.md experiment plan in parallel.")
    parser.add_argument(
        "--plan",
        choices=["crossplay", "ablation"],
        default="crossplay",
        help="Which predefined plan to run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory used to store all planned experiment outputs. Defaults depend on --plan.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260414,
        help="Base integer used to generate unique per-game seeds for every job.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker processes. Defaults to a plan-sized heuristic.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to launch experiments/run_experiments.py.",
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        default=None,
        help="Optional subset of job keys to run within the selected plan.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs whose output directory already contains summary.csv and per_game.csv.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the jobs and generated seeds without launching them.",
    )
    return parser.parse_args()


def default_output_root_for_plan(plan: str) -> Path:
    if plan == "crossplay":
        return CROSSPLAY_OUTPUT_ROOT
    if plan == "ablation":
        return ABLATION_OUTPUT_ROOT
    raise ValueError(f"Unsupported plan: {plan!r}")


def plan_jobs(plan: str):
    if plan == "crossplay":
        return CROSSPLAY_JOBS
    if plan == "ablation":
        return ABLATION_JOBS
    raise ValueError(f"Unsupported plan: {plan!r}")


def plan_groups(plan: str):
    if plan == "crossplay":
        return CROSSPLAY_GROUPS
    if plan == "ablation":
        return ABLATION_GROUPS
    raise ValueError(f"Unsupported plan: {plan!r}")


def ensure_output_root(path: Path | None, plan: str) -> Path:
    if path is None:
        return default_output_root_for_plan(plan)
    if not path.is_absolute():
        return REPO_ROOT / path
    return path


def default_max_workers(job_count: int) -> int:
    return max(1, min(job_count, (os.cpu_count() or 2) // 2 or 1))


def build_seed_list(job_index: int, job, base_seed: int) -> list[int]:
    start = base_seed + job_index * 1000
    return [start + offset for offset in range(job.games)]


def build_job_output_dir(output_root: Path, job) -> Path:
    return output_root / job.key


def split_job_seeds(job, seeds: list[int]) -> tuple[list[int], list[int]]:
    if len(seeds) != job.games:
        raise ValueError(
            f"Seed count mismatch for {job.key}: expected {job.games}, got {len(seeds)}."
        )
    if job.games % 2 != 0:
        raise ValueError(f"{job.key} requires an even number of games, got {job.games}.")
    midpoint = job.games // 2
    return seeds[:midpoint], seeds[midpoint:]


def build_subrun_output_dir(output_dir: Path, prefix: str, color: str) -> Path:
    return output_dir / f"{prefix}_{color}"


def build_crossplay_command(
    python_exe: str,
    job: CrossplayJobSpec,
    seeds: list[int],
    output_dir: Path,
    mcts_color: str,
) -> list[str]:
    if mcts_color == "black":
        black_agent = "mcts"
        white_agent = "minimax"
    elif mcts_color == "white":
        black_agent = "minimax"
        white_agent = "mcts"
    else:
        raise ValueError(f"Unsupported mcts_color: {mcts_color!r}")

    return [
        python_exe,
        str(RUN_EXPERIMENTS),
        "--black-agent",
        black_agent,
        "--white-agent",
        white_agent,
        "--mcts-rounds",
        str(job.mcts_rounds),
        "--minimax-depth",
        str(job.minimax_depth),
        "--games",
        str(len(seeds)),
        "--seeds",
        *[str(seed) for seed in seeds],
        "--output-dir",
        str(output_dir),
    ]


def build_ablation_command(
    python_exe: str,
    job: AblationJobSpec,
    seeds: list[int],
    output_dir: Path,
    standard_color: str,
) -> list[str]:
    baseline_config = job.resolved_baseline_config()
    if standard_color == "black":
        black_config = baseline_config
        white_config = job.variant_config
    elif standard_color == "white":
        black_config = job.variant_config
        white_config = baseline_config
    else:
        raise ValueError(f"Unsupported standard_color: {standard_color!r}")

    return [
        python_exe,
        str(RUN_EXPERIMENTS),
        "--black-agent",
        "mcts",
        "--white-agent",
        "mcts",
        *black_config.to_cli_args("black"),
        *white_config.to_cli_args("white"),
        "--games",
        str(len(seeds)),
        "--seeds",
        *[str(seed) for seed in seeds],
        "--output-dir",
        str(output_dir),
    ]


def should_skip_job(output_dir: Path) -> bool:
    return (output_dir / "summary.csv").exists() and (output_dir / "per_game.csv").exists()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_job_index(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "job_key",
        "plan",
        "primary_label",
        "secondary_label",
        "games",
        "output_dir",
        "status",
        "duration_s",
        "seed_list",
        "note",
    ]
    write_csv(path, rows, fieldnames)


def write_manifest(path: Path, payload: dict) -> None:
    def _json_default(value):
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def run_job(command: list[str], workdir: str) -> dict:
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    duration_s = time.perf_counter() - start
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "duration_s": duration_s,
    }


def summarize_crossplay_rows(
    rows: list[dict[str, str]],
    *,
    job: CrossplayJobSpec,
    move_time_limit_s: float,
) -> dict:
    if not rows:
        raise ValueError(f"No per-game rows available for {job.key}.")

    terminations = Counter(row["termination_reason"] for row in rows)
    games = len(rows)
    mcts_wins = sum(int(row["mcts_won"]) for row in rows)
    minimax_wins = sum(int(row["minimax_won"]) for row in rows)

    return {
        "mcts_label": f"MCTS(rounds={job.mcts_rounds})",
        "minimax_label": f"Minimax(depth={job.minimax_depth})",
        "mcts_rounds": job.mcts_rounds,
        "minimax_depth": job.minimax_depth,
        "games": games,
        "games_mcts_black": sum(1 for row in rows if row["mcts_color"] == "black"),
        "games_mcts_white": sum(1 for row in rows if row["mcts_color"] == "white"),
        "mcts_win_rate": mcts_wins / games,
        "minimax_win_rate": minimax_wins / games,
        "mcts_win_rate_percent": 100.0 * mcts_wins / games,
        "minimax_win_rate_percent": 100.0 * minimax_wins / games,
        "avg_total_duration_s": sum(float(row["total_duration_s"]) for row in rows) / games,
        "avg_move_count": sum(float(row["move_count"]) for row in rows) / games,
        "avg_mcts_move_time_s": sum(float(row["mcts_avg_move_time_s"]) for row in rows) / games,
        "avg_minimax_move_time_s": sum(float(row["minimax_avg_move_time_s"]) for row in rows)
        / games,
        "max_mcts_move_time_s": max(float(row["mcts_max_move_time_s"]) for row in rows),
        "max_minimax_move_time_s": max(float(row["minimax_max_move_time_s"]) for row in rows),
        "mcts_exceeded_move_time_limit_games": sum(
            int(row["mcts_exceeded_move_time_limit"]) for row in rows
        ),
        "minimax_exceeded_move_time_limit_games": sum(
            int(row["minimax_exceeded_move_time_limit"]) for row in rows
        ),
        "move_time_limit_s": move_time_limit_s,
        "avg_mcts_margin": sum(float(row["mcts_margin"]) for row in rows) / games,
        "avg_abs_margin": sum(float(row["abs_margin"]) for row in rows) / games,
        "termination_summary": "; ".join(
            f"{key}:{terminations[key]}" for key in sorted(terminations.keys())
        ),
    }


def merge_crossplay_job_outputs(
    job: CrossplayJobSpec,
    output_dir: Path,
    seeds: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, str | int]] = []
    global_game_index = 0
    move_time_limit_s: float | None = None

    for mcts_color in ("black", "white"):
        subrun_name = f"mcts_{mcts_color}"
        subrun_dir = build_subrun_output_dir(output_dir, "mcts", mcts_color)
        per_game_rows = read_csv_rows(subrun_dir / "per_game.csv")
        summary_rows = read_csv_rows(subrun_dir / "summary.csv")
        if not summary_rows:
            raise ValueError(f"Missing summary row for {job.key} subrun {subrun_name}.")

        subrun_move_time_limit = float(summary_rows[0]["move_time_limit_s"])
        if move_time_limit_s is None:
            move_time_limit_s = subrun_move_time_limit
        elif abs(move_time_limit_s - subrun_move_time_limit) > 1e-9:
            raise ValueError(
                f"Inconsistent move time limit in {job.key}: "
                f"{move_time_limit_s} vs {subrun_move_time_limit}."
            )

        for row in per_game_rows:
            mcts_is_black = mcts_color == "black"
            mcts_won = int(row["black_won"]) if mcts_is_black else int(row["white_won"])
            minimax_won = int(row["white_won"]) if mcts_is_black else int(row["black_won"])
            mcts_turns = int(row["black_turns"]) if mcts_is_black else int(row["white_turns"])
            minimax_turns = (
                int(row["white_turns"]) if mcts_is_black else int(row["black_turns"])
            )
            mcts_think_time_s = (
                float(row["black_think_time_s"])
                if mcts_is_black
                else float(row["white_think_time_s"])
            )
            minimax_think_time_s = (
                float(row["white_think_time_s"])
                if mcts_is_black
                else float(row["black_think_time_s"])
            )
            mcts_avg_move_time_s = (
                float(row["black_avg_move_time_s"])
                if mcts_is_black
                else float(row["white_avg_move_time_s"])
            )
            minimax_avg_move_time_s = (
                float(row["white_avg_move_time_s"])
                if mcts_is_black
                else float(row["black_avg_move_time_s"])
            )
            mcts_max_move_time_s = (
                float(row["black_max_move_time_s"])
                if mcts_is_black
                else float(row["white_max_move_time_s"])
            )
            minimax_max_move_time_s = (
                float(row["white_max_move_time_s"])
                if mcts_is_black
                else float(row["black_max_move_time_s"])
            )
            mcts_exceeded_move_time_limit = (
                int(row["black_exceeded_move_time_limit"])
                if mcts_is_black
                else int(row["white_exceeded_move_time_limit"])
            )
            minimax_exceeded_move_time_limit = (
                int(row["white_exceeded_move_time_limit"])
                if mcts_is_black
                else int(row["black_exceeded_move_time_limit"])
            )
            black_margin = float(row["black_margin"])
            mcts_margin = black_margin if mcts_is_black else -black_margin

            merged_rows.append(
                {
                    "game_index": global_game_index,
                    "subrun": subrun_name,
                    "subrun_game_index": row["game_index"],
                    "seed": row["seed"],
                    "mcts_color": mcts_color,
                    "minimax_color": "white" if mcts_is_black else "black",
                    "mcts_label": f"MCTS(rounds={job.mcts_rounds})",
                    "minimax_label": f"Minimax(depth={job.minimax_depth})",
                    "winner_color": row["winner_color"],
                    "winner_label": row["winner_label"],
                    "mcts_won": mcts_won,
                    "minimax_won": minimax_won,
                    "move_count": row["move_count"],
                    "termination_reason": row["termination_reason"],
                    "total_duration_s": row["total_duration_s"],
                    "mcts_turns": mcts_turns,
                    "minimax_turns": minimax_turns,
                    "mcts_think_time_s": f"{mcts_think_time_s:.12g}",
                    "minimax_think_time_s": f"{minimax_think_time_s:.12g}",
                    "mcts_avg_move_time_s": f"{mcts_avg_move_time_s:.12g}",
                    "minimax_avg_move_time_s": f"{minimax_avg_move_time_s:.12g}",
                    "mcts_max_move_time_s": f"{mcts_max_move_time_s:.12g}",
                    "minimax_max_move_time_s": f"{minimax_max_move_time_s:.12g}",
                    "mcts_exceeded_move_time_limit": mcts_exceeded_move_time_limit,
                    "minimax_exceeded_move_time_limit": minimax_exceeded_move_time_limit,
                    "mcts_margin": f"{mcts_margin:.12g}",
                    "abs_margin": f"{abs(mcts_margin):.12g}",
                    "board_size": row["board_size"],
                    "move_limit": row["move_limit"],
                    "komi": row["komi"],
                    "black_label": row["black_label"],
                    "white_label": row["white_label"],
                    "black_family": row["black_family"],
                    "white_family": row["white_family"],
                    "black_won": row["black_won"],
                    "white_won": row["white_won"],
                    "black_margin": row["black_margin"],
                    "score_b": row["score_b"],
                    "score_w": row["score_w"],
                    "result_text": row["result_text"],
                }
            )
            global_game_index += 1

    if move_time_limit_s is None:
        raise ValueError(f"No move time limit resolved for {job.key}.")

    summary_row = summarize_crossplay_rows(
        merged_rows,
        job=job,
        move_time_limit_s=move_time_limit_s,
    )

    per_game_fieldnames = [
        "game_index",
        "subrun",
        "subrun_game_index",
        "seed",
        "mcts_color",
        "minimax_color",
        "mcts_label",
        "minimax_label",
        "winner_color",
        "winner_label",
        "mcts_won",
        "minimax_won",
        "move_count",
        "termination_reason",
        "total_duration_s",
        "mcts_turns",
        "minimax_turns",
        "mcts_think_time_s",
        "minimax_think_time_s",
        "mcts_avg_move_time_s",
        "minimax_avg_move_time_s",
        "mcts_max_move_time_s",
        "minimax_max_move_time_s",
        "mcts_exceeded_move_time_limit",
        "minimax_exceeded_move_time_limit",
        "mcts_margin",
        "abs_margin",
        "board_size",
        "move_limit",
        "komi",
        "black_label",
        "white_label",
        "black_family",
        "white_family",
        "black_won",
        "white_won",
        "black_margin",
        "score_b",
        "score_w",
        "result_text",
    ]
    summary_fieldnames = [
        "mcts_label",
        "minimax_label",
        "mcts_rounds",
        "minimax_depth",
        "games",
        "games_mcts_black",
        "games_mcts_white",
        "mcts_win_rate",
        "minimax_win_rate",
        "mcts_win_rate_percent",
        "minimax_win_rate_percent",
        "avg_total_duration_s",
        "avg_move_count",
        "avg_mcts_move_time_s",
        "avg_minimax_move_time_s",
        "max_mcts_move_time_s",
        "max_minimax_move_time_s",
        "mcts_exceeded_move_time_limit_games",
        "minimax_exceeded_move_time_limit_games",
        "move_time_limit_s",
        "avg_mcts_margin",
        "avg_abs_margin",
        "termination_summary",
    ]

    write_csv(output_dir / "per_game.csv", merged_rows, per_game_fieldnames)
    write_csv(output_dir / "summary.csv", [summary_row], summary_fieldnames)
    write_csv(
        output_dir / "game_seeds.csv",
        [
            {
                "game_index": row["game_index"],
                "seed": row["seed"],
                "subrun": row["subrun"],
                "subrun_game_index": row["subrun_game_index"],
                "mcts_color": row["mcts_color"],
            }
            for row in merged_rows
        ],
        ["game_index", "seed", "subrun", "subrun_game_index", "mcts_color"],
    )
    write_manifest(
        output_dir / "summary.json",
        {
            "metadata": {
                "plan": "crossplay",
                "job_key": job.key,
                "mcts_rounds": job.mcts_rounds,
                "minimax_depth": job.minimax_depth,
                "games": job.games,
                "games_per_side": job.games // 2,
                "seeds": seeds,
                "output_dir": str(output_dir),
                "subruns": [
                    {
                        "name": "mcts_black",
                        "mcts_color": "black",
                        "output_dir": str(build_subrun_output_dir(output_dir, "mcts", "black")),
                    },
                    {
                        "name": "mcts_white",
                        "mcts_color": "white",
                        "output_dir": str(build_subrun_output_dir(output_dir, "mcts", "white")),
                    },
                ],
            },
            "summary": summary_row,
            "per_game_count": len(merged_rows),
        },
    )


def summarize_ablation_rows(
    rows: list[dict[str, str]],
    *,
    job: AblationJobSpec,
    move_time_limit_s: float,
) -> dict:
    if not rows:
        raise ValueError(f"No per-game rows available for {job.key}.")

    terminations = Counter(row["termination_reason"] for row in rows)
    games = len(rows)
    variant_wins = sum(int(row["variant_won"]) for row in rows)
    standard_wins = sum(int(row["standard_won"]) for row in rows)

    return {
        "variant_name": job.variant_name,
        "variant_label": job.variant_config.label(),
        "standard_name": job.baseline_name,
        "standard_label": job.resolved_baseline_config().label(),
        "games": games,
        "games_standard_black": sum(1 for row in rows if row["standard_color"] == "black"),
        "games_standard_white": sum(1 for row in rows if row["standard_color"] == "white"),
        "variant_win_rate": variant_wins / games,
        "standard_win_rate": standard_wins / games,
        "variant_win_rate_percent": 100.0 * variant_wins / games,
        "standard_win_rate_percent": 100.0 * standard_wins / games,
        "avg_total_duration_s": sum(float(row["total_duration_s"]) for row in rows) / games,
        "avg_move_count": sum(float(row["move_count"]) for row in rows) / games,
        "avg_variant_move_time_s": sum(float(row["variant_avg_move_time_s"]) for row in rows)
        / games,
        "avg_standard_move_time_s": sum(float(row["standard_avg_move_time_s"]) for row in rows)
        / games,
        "max_variant_move_time_s": max(float(row["variant_max_move_time_s"]) for row in rows),
        "max_standard_move_time_s": max(
            float(row["standard_max_move_time_s"]) for row in rows
        ),
        "variant_exceeded_move_time_limit_games": sum(
            int(row["variant_exceeded_move_time_limit"]) for row in rows
        ),
        "standard_exceeded_move_time_limit_games": sum(
            int(row["standard_exceeded_move_time_limit"]) for row in rows
        ),
        "move_time_limit_s": move_time_limit_s,
        "avg_variant_margin": sum(float(row["variant_margin"]) for row in rows) / games,
        "avg_abs_margin": sum(float(row["abs_margin"]) for row in rows) / games,
        "termination_summary": "; ".join(
            f"{key}:{terminations[key]}" for key in sorted(terminations.keys())
        ),
    }


def merge_ablation_job_outputs(
    job: AblationJobSpec,
    output_dir: Path,
    seeds: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, str | int]] = []
    global_game_index = 0
    move_time_limit_s: float | None = None

    for standard_color in ("black", "white"):
        subrun_name = f"standard_{standard_color}"
        subrun_dir = build_subrun_output_dir(output_dir, "standard", standard_color)
        per_game_rows = read_csv_rows(subrun_dir / "per_game.csv")
        summary_rows = read_csv_rows(subrun_dir / "summary.csv")
        if not summary_rows:
            raise ValueError(f"Missing summary row for {job.key} subrun {subrun_name}.")

        subrun_move_time_limit = float(summary_rows[0]["move_time_limit_s"])
        if move_time_limit_s is None:
            move_time_limit_s = subrun_move_time_limit
        elif abs(move_time_limit_s - subrun_move_time_limit) > 1e-9:
            raise ValueError(
                f"Inconsistent move time limit in {job.key}: "
                f"{move_time_limit_s} vs {subrun_move_time_limit}."
            )

        for row in per_game_rows:
            standard_is_black = standard_color == "black"
            variant_won = int(row["white_won"]) if standard_is_black else int(row["black_won"])
            standard_won = int(row["black_won"]) if standard_is_black else int(row["white_won"])
            variant_turns = int(row["white_turns"]) if standard_is_black else int(row["black_turns"])
            standard_turns = (
                int(row["black_turns"]) if standard_is_black else int(row["white_turns"])
            )
            variant_think_time_s = (
                float(row["white_think_time_s"])
                if standard_is_black
                else float(row["black_think_time_s"])
            )
            standard_think_time_s = (
                float(row["black_think_time_s"])
                if standard_is_black
                else float(row["white_think_time_s"])
            )
            variant_avg_move_time_s = (
                float(row["white_avg_move_time_s"])
                if standard_is_black
                else float(row["black_avg_move_time_s"])
            )
            standard_avg_move_time_s = (
                float(row["black_avg_move_time_s"])
                if standard_is_black
                else float(row["white_avg_move_time_s"])
            )
            variant_max_move_time_s = (
                float(row["white_max_move_time_s"])
                if standard_is_black
                else float(row["black_max_move_time_s"])
            )
            standard_max_move_time_s = (
                float(row["black_max_move_time_s"])
                if standard_is_black
                else float(row["white_max_move_time_s"])
            )
            variant_exceeded_move_time_limit = (
                int(row["white_exceeded_move_time_limit"])
                if standard_is_black
                else int(row["black_exceeded_move_time_limit"])
            )
            standard_exceeded_move_time_limit = (
                int(row["black_exceeded_move_time_limit"])
                if standard_is_black
                else int(row["white_exceeded_move_time_limit"])
            )
            black_margin = float(row["black_margin"])
            variant_margin = -black_margin if standard_is_black else black_margin

            merged_rows.append(
                {
                    "game_index": global_game_index,
                    "subrun": subrun_name,
                    "subrun_game_index": row["game_index"],
                    "seed": row["seed"],
                    "variant_color": "white" if standard_is_black else "black",
                    "standard_color": standard_color,
                    "variant_name": job.variant_name,
                    "variant_label": job.variant_config.label(),
                    "standard_name": job.baseline_name,
                    "standard_label": job.resolved_baseline_config().label(),
                    "winner_color": row["winner_color"],
                    "winner_label": row["winner_label"],
                    "variant_won": variant_won,
                    "standard_won": standard_won,
                    "move_count": row["move_count"],
                    "termination_reason": row["termination_reason"],
                    "total_duration_s": row["total_duration_s"],
                    "variant_turns": variant_turns,
                    "standard_turns": standard_turns,
                    "variant_think_time_s": f"{variant_think_time_s:.12g}",
                    "standard_think_time_s": f"{standard_think_time_s:.12g}",
                    "variant_avg_move_time_s": f"{variant_avg_move_time_s:.12g}",
                    "standard_avg_move_time_s": f"{standard_avg_move_time_s:.12g}",
                    "variant_max_move_time_s": f"{variant_max_move_time_s:.12g}",
                    "standard_max_move_time_s": f"{standard_max_move_time_s:.12g}",
                    "variant_exceeded_move_time_limit": variant_exceeded_move_time_limit,
                    "standard_exceeded_move_time_limit": standard_exceeded_move_time_limit,
                    "variant_margin": f"{variant_margin:.12g}",
                    "abs_margin": f"{abs(variant_margin):.12g}",
                    "board_size": row["board_size"],
                    "move_limit": row["move_limit"],
                    "komi": row["komi"],
                    "black_label": row["black_label"],
                    "white_label": row["white_label"],
                    "black_family": row["black_family"],
                    "white_family": row["white_family"],
                    "black_won": row["black_won"],
                    "white_won": row["white_won"],
                    "black_margin": row["black_margin"],
                    "score_b": row["score_b"],
                    "score_w": row["score_w"],
                    "result_text": row["result_text"],
                }
            )
            global_game_index += 1

    if move_time_limit_s is None:
        raise ValueError(f"No move time limit resolved for {job.key}.")

    summary_row = summarize_ablation_rows(
        merged_rows,
        job=job,
        move_time_limit_s=move_time_limit_s,
    )

    per_game_fieldnames = [
        "game_index",
        "subrun",
        "subrun_game_index",
        "seed",
        "variant_color",
        "standard_color",
        "variant_name",
        "variant_label",
        "standard_name",
        "standard_label",
        "winner_color",
        "winner_label",
        "variant_won",
        "standard_won",
        "move_count",
        "termination_reason",
        "total_duration_s",
        "variant_turns",
        "standard_turns",
        "variant_think_time_s",
        "standard_think_time_s",
        "variant_avg_move_time_s",
        "standard_avg_move_time_s",
        "variant_max_move_time_s",
        "standard_max_move_time_s",
        "variant_exceeded_move_time_limit",
        "standard_exceeded_move_time_limit",
        "variant_margin",
        "abs_margin",
        "board_size",
        "move_limit",
        "komi",
        "black_label",
        "white_label",
        "black_family",
        "white_family",
        "black_won",
        "white_won",
        "black_margin",
        "score_b",
        "score_w",
        "result_text",
    ]
    summary_fieldnames = [
        "variant_name",
        "variant_label",
        "standard_name",
        "standard_label",
        "games",
        "games_standard_black",
        "games_standard_white",
        "variant_win_rate",
        "standard_win_rate",
        "variant_win_rate_percent",
        "standard_win_rate_percent",
        "avg_total_duration_s",
        "avg_move_count",
        "avg_variant_move_time_s",
        "avg_standard_move_time_s",
        "max_variant_move_time_s",
        "max_standard_move_time_s",
        "variant_exceeded_move_time_limit_games",
        "standard_exceeded_move_time_limit_games",
        "move_time_limit_s",
        "avg_variant_margin",
        "avg_abs_margin",
        "termination_summary",
    ]

    write_csv(output_dir / "per_game.csv", merged_rows, per_game_fieldnames)
    write_csv(output_dir / "summary.csv", [summary_row], summary_fieldnames)
    write_csv(
        output_dir / "game_seeds.csv",
        [
            {
                "game_index": row["game_index"],
                "seed": row["seed"],
                "subrun": row["subrun"],
                "subrun_game_index": row["subrun_game_index"],
                "standard_color": row["standard_color"],
            }
            for row in merged_rows
        ],
        ["game_index", "seed", "subrun", "subrun_game_index", "standard_color"],
    )
    write_manifest(
        output_dir / "summary.json",
        {
            "metadata": {
                "plan": "ablation",
                "job_key": job.key,
                "variant_name": job.variant_name,
                "variant_config": asdict(job.variant_config),
                "baseline_name": job.baseline_name,
                "baseline_config": asdict(job.resolved_baseline_config()),
                "games": job.games,
                "games_per_side": job.games // 2,
                "seeds": seeds,
                "output_dir": str(output_dir),
                "subruns": [
                    {
                        "name": "standard_black",
                        "standard_color": "black",
                        "output_dir": str(
                            build_subrun_output_dir(output_dir, "standard", "black")
                        ),
                    },
                    {
                        "name": "standard_white",
                        "standard_color": "white",
                        "output_dir": str(
                            build_subrun_output_dir(output_dir, "standard", "white")
                        ),
                    },
                ],
            },
            "summary": summary_row,
            "per_game_count": len(merged_rows),
        },
    )


def run_crossplay_job_bundle(
    job: CrossplayJobSpec,
    commands: list[dict],
    output_dir: Path,
    seeds: list[int],
) -> dict:
    total_start = time.perf_counter()
    subruns = []

    for command_spec in commands:
        result = run_job(command_spec["command"], str(REPO_ROOT))
        subruns.append(
            {
                "name": command_spec["name"],
                "mcts_color": command_spec["mcts_color"],
                "output_dir": str(command_spec["output_dir"]),
                "command": command_spec["command"],
                **result,
            }
        )
        if result["returncode"] != 0:
            return {
                "returncode": result["returncode"],
                "duration_s": time.perf_counter() - total_start,
                "subruns": subruns,
            }

    merge_crossplay_job_outputs(job, output_dir, seeds)
    return {
        "returncode": 0,
        "duration_s": time.perf_counter() - total_start,
        "subruns": subruns,
    }


def run_ablation_job_bundle(
    job: AblationJobSpec,
    commands: list[dict],
    output_dir: Path,
    seeds: list[int],
) -> dict:
    total_start = time.perf_counter()
    subruns = []

    for command_spec in commands:
        result = run_job(command_spec["command"], str(REPO_ROOT))
        subruns.append(
            {
                "name": command_spec["name"],
                "standard_color": command_spec["standard_color"],
                "output_dir": str(command_spec["output_dir"]),
                "command": command_spec["command"],
                **result,
            }
        )
        if result["returncode"] != 0:
            return {
                "returncode": result["returncode"],
                "duration_s": time.perf_counter() - total_start,
                "subruns": subruns,
            }

    merge_ablation_job_outputs(job, output_dir, seeds)
    return {
        "returncode": 0,
        "duration_s": time.perf_counter() - total_start,
        "subruns": subruns,
    }


def main() -> None:
    args = parse_args()
    jobs = plan_jobs(args.plan)
    groups = plan_groups(args.plan)
    output_root = ensure_output_root(args.output_root, args.plan)
    output_root.mkdir(parents=True, exist_ok=True)

    selected_job_keys = set(args.jobs) if args.jobs is not None else None
    scheduled_jobs = []
    job_index_rows = []

    for job_index, job in enumerate(jobs):
        if selected_job_keys is not None and job.key not in selected_job_keys:
            continue

        seeds = build_seed_list(job_index, job, args.base_seed)
        output_dir = build_job_output_dir(output_root, job)
        black_seeds, white_seeds = split_job_seeds(job, seeds)

        if args.plan == "crossplay":
            commands = [
                {
                    "name": "mcts_black",
                    "mcts_color": "black",
                    "output_dir": build_subrun_output_dir(output_dir, "mcts", "black"),
                    "command": build_crossplay_command(
                        args.python,
                        job,
                        black_seeds,
                        build_subrun_output_dir(output_dir, "mcts", "black"),
                        "black",
                    ),
                },
                {
                    "name": "mcts_white",
                    "mcts_color": "white",
                    "output_dir": build_subrun_output_dir(output_dir, "mcts", "white"),
                    "command": build_crossplay_command(
                        args.python,
                        job,
                        white_seeds,
                        build_subrun_output_dir(output_dir, "mcts", "white"),
                        "white",
                    ),
                },
            ]
            primary_label = f"MCTS(rounds={job.mcts_rounds})"
            secondary_label = f"Minimax(depth={job.minimax_depth})"
            runner = run_crossplay_job_bundle
        else:
            commands = [
                {
                    "name": "standard_black",
                    "standard_color": "black",
                    "output_dir": build_subrun_output_dir(output_dir, "standard", "black"),
                    "command": build_ablation_command(
                        args.python,
                        job,
                        black_seeds,
                        build_subrun_output_dir(output_dir, "standard", "black"),
                        "black",
                    ),
                },
                {
                    "name": "standard_white",
                    "standard_color": "white",
                    "output_dir": build_subrun_output_dir(output_dir, "standard", "white"),
                    "command": build_ablation_command(
                        args.python,
                        job,
                        white_seeds,
                        build_subrun_output_dir(output_dir, "standard", "white"),
                        "white",
                    ),
                },
            ]
            primary_label = job.variant_name
            secondary_label = job.baseline_name
            runner = run_ablation_job_bundle

        skip = args.skip_existing and should_skip_job(output_dir)
        scheduled_jobs.append(
            {
                "job": job,
                "plan": args.plan,
                "seeds": seeds,
                "output_dir": output_dir,
                "commands": commands,
                "skip": skip,
                "runner": runner,
            }
        )
        job_index_rows.append(
            {
                "job_key": job.key,
                "plan": args.plan,
                "primary_label": primary_label,
                "secondary_label": secondary_label,
                "games": job.games,
                "output_dir": str(output_dir),
                "status": "pending" if not skip else "skipped_existing",
                "duration_s": "",
                "seed_list": " ".join(str(seed) for seed in seeds),
                "note": job.note,
            }
        )

    if selected_job_keys is not None:
        missing_jobs = sorted(selected_job_keys - {item["job"].key for item in scheduled_jobs})
        if missing_jobs:
            raise ValueError(f"Unknown job keys for plan {args.plan}: {', '.join(missing_jobs)}")

    if args.dry_run:
        for item in scheduled_jobs:
            print(item["job"].key)
            print("  plan:", item["plan"])
            print("  seeds:", " ".join(str(seed) for seed in item["seeds"]))
            print("  output:", item["output_dir"])
            for command_spec in item["commands"]:
                color_key = "mcts_color" if "mcts_color" in command_spec else "standard_color"
                print(
                    f"  subrun[{command_spec['name']}] ({color_key}={command_spec[color_key]}):",
                    " ".join(command_spec["command"]),
                )
        return

    max_workers = args.max_workers or default_max_workers(len(scheduled_jobs))
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "plan": args.plan,
        "python": args.python,
        "base_seed": args.base_seed,
        "max_workers": max_workers,
        "output_root": str(output_root),
        "groups": [asdict(group) for group in groups],
        "jobs": [],
    }

    future_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for item in scheduled_jobs:
            job = item["job"]
            if item["skip"]:
                manifest["jobs"].append(
                    {
                        "plan": args.plan,
                        "job": asdict(job),
                        "seeds": item["seeds"],
                        "output_dir": str(item["output_dir"]),
                        "status": "skipped_existing",
                        "duration_s": 0.0,
                    }
                )
                continue

            future = executor.submit(
                item["runner"],
                job,
                item["commands"],
                item["output_dir"],
                item["seeds"],
            )
            future_map[future] = item

        for future in as_completed(future_map):
            item = future_map[future]
            job = item["job"]
            result = future.result()
            status = "completed" if result["returncode"] == 0 else "failed"
            manifest["jobs"].append(
                {
                    "plan": args.plan,
                    "job": asdict(job),
                    "seeds": item["seeds"],
                    "output_dir": str(item["output_dir"]),
                    "status": status,
                    "duration_s": result["duration_s"],
                    "returncode": result["returncode"],
                    "subruns": result["subruns"],
                }
            )

            for row in job_index_rows:
                if row["job_key"] != job.key:
                    continue
                row["status"] = status
                row["duration_s"] = f"{result['duration_s']:.3f}"
                break

            print(f"[{status}] {job.key} ({result['duration_s']:.2f}s)")
            if result["returncode"] != 0:
                for subrun in result["subruns"]:
                    if subrun["returncode"] != 0 and subrun["stderr"]:
                        print(subrun["stderr"])

    write_job_index(output_root / "job_index.csv", job_index_rows)
    write_manifest(output_root / "manifest.json", manifest)

    failed_jobs = [job for job in manifest["jobs"] if job["status"] == "failed"]
    print(f"Saved job index to {output_root / 'job_index.csv'}")
    print(f"Saved manifest to {output_root / 'manifest.json'}")
    if failed_jobs:
        raise SystemExit(f"{len(failed_jobs)} job(s) failed.")


if __name__ == "__main__":
    main()
