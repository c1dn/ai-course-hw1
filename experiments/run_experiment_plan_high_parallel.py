"""
Higher-parallelism launcher for experiment plans.

Compared with ``run_experiment_plan.py``, this version schedules each
game as its own subprocess. Once all 10 games for a job finish, it
rebuilds the per-subrun CSVs and then reuses the existing merge logic to
produce the same job-level outputs.
"""

from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import run_experiment_plan as base
import run_experiments as runner


@dataclass(frozen=True)
class GameTaskSpec:
    job_key: str
    job_index: int
    plan: str
    subrun_name: str
    subrun_dir: Path
    game_index_within_subrun: int
    global_seed_index: int
    seed: int
    command: list[str]
    output_dir: Path
    color_key: str
    color_value: str


def read_csv_rows_and_fieldnames(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def coerce_per_game_rows_for_summary(rows: list[dict[str, str]]) -> list[dict]:
    converted = []
    for row in rows:
        converted.append(
            {
                **row,
                "black_won": int(row["black_won"]),
                "white_won": int(row["white_won"]),
                "move_count": float(row["move_count"]),
                "total_duration_s": float(row["total_duration_s"]),
                "black_avg_move_time_s": float(row["black_avg_move_time_s"]),
                "white_avg_move_time_s": float(row["white_avg_move_time_s"]),
                "black_max_move_time_s": float(row["black_max_move_time_s"]),
                "white_max_move_time_s": float(row["white_max_move_time_s"]),
                "black_exceeded_move_time_limit": int(row["black_exceeded_move_time_limit"]),
                "white_exceeded_move_time_limit": int(row["white_exceeded_move_time_limit"]),
                "black_margin": float(row["black_margin"]),
            }
        )
    return converted


def aggregate_single_game_outputs(
    *,
    subrun_dir: Path,
    game_tasks: list[GameTaskSpec],
) -> dict:
    ordered_tasks = sorted(game_tasks, key=lambda task: task.game_index_within_subrun)
    merged_rows: list[dict[str, str]] = []
    per_game_fieldnames: list[str] | None = None
    summary_fieldnames: list[str] | None = None
    move_time_limit_s: float | None = None
    first_summary_payload: dict | None = None

    for subrun_game_index, task in enumerate(ordered_tasks):
        game_per_game_rows, current_per_game_fieldnames = read_csv_rows_and_fieldnames(
            task.output_dir / "per_game.csv"
        )
        if len(game_per_game_rows) != 1:
            raise ValueError(
                f"Expected exactly 1 per-game row in {task.output_dir}, got {len(game_per_game_rows)}."
            )
        game_summary_rows, current_summary_fieldnames = read_csv_rows_and_fieldnames(
            task.output_dir / "summary.csv"
        )
        if len(game_summary_rows) != 1:
            raise ValueError(
                f"Expected exactly 1 summary row in {task.output_dir}, got {len(game_summary_rows)}."
            )

        if per_game_fieldnames is None:
            per_game_fieldnames = current_per_game_fieldnames
        if summary_fieldnames is None:
            summary_fieldnames = current_summary_fieldnames

        summary_row = game_summary_rows[0]
        current_move_time_limit = float(summary_row["move_time_limit_s"])
        if move_time_limit_s is None:
            move_time_limit_s = current_move_time_limit
        elif abs(move_time_limit_s - current_move_time_limit) > 1e-9:
            raise ValueError(
                f"Inconsistent move_time_limit_s in {subrun_dir}: "
                f"{move_time_limit_s} vs {current_move_time_limit}."
            )

        if first_summary_payload is None:
            with (task.output_dir / "summary.json").open("r", encoding="utf-8") as handle:
                first_summary_payload = json.load(handle)

        row = dict(game_per_game_rows[0])
        row["game_index"] = str(subrun_game_index)
        merged_rows.append(row)

    if not merged_rows or per_game_fieldnames is None or summary_fieldnames is None:
        raise ValueError(f"No game outputs found for {subrun_dir}.")
    if move_time_limit_s is None or first_summary_payload is None:
        raise ValueError(f"Failed to resolve summary metadata for {subrun_dir}.")

    first_row = merged_rows[0]
    black_spec = SimpleNamespace(
        label=first_row["black_label"],
        family=first_row["black_family"],
    )
    white_spec = SimpleNamespace(
        label=first_row["white_label"],
        family=first_row["white_family"],
    )
    summary_row = runner.summarize_rows(
        coerce_per_game_rows_for_summary(merged_rows),
        black_spec=black_spec,
        white_spec=white_spec,
        move_time_limit_s=move_time_limit_s,
    )

    subrun_dir.mkdir(parents=True, exist_ok=True)
    base.write_csv(subrun_dir / "per_game.csv", merged_rows, per_game_fieldnames)
    base.write_csv(subrun_dir / "summary.csv", [summary_row], summary_fieldnames)
    base.write_csv(
        subrun_dir / "game_seeds.csv",
        [
            {"game_index": index, "seed": task.seed}
            for index, task in enumerate(ordered_tasks)
        ],
        ["game_index", "seed"],
    )

    metadata = dict(first_summary_payload["metadata"])
    metadata["games"] = len(ordered_tasks)
    metadata["seeds"] = [task.seed for task in ordered_tasks]
    metadata["seed_base"] = ordered_tasks[0].seed
    metadata["output_dir"] = str(subrun_dir)
    base.write_manifest(
        subrun_dir / "summary.json",
        {
            "metadata": metadata,
            "summary": summary_row,
            "per_game_count": len(merged_rows),
        },
    )
    return summary_row


def build_job_game_tasks(args, job, job_index: int, output_dir: Path, seeds: list[int]) -> list[GameTaskSpec]:
    black_seeds, white_seeds = base.split_job_seeds(job, seeds)
    tasks: list[GameTaskSpec] = []

    if args.plan == "crossplay":
        subruns = [
            ("mcts_black", "mcts_color", "black", black_seeds),
            ("mcts_white", "mcts_color", "white", white_seeds),
        ]
        for subrun_name, color_key, color_value, subrun_seeds in subruns:
            subrun_dir = base.build_subrun_output_dir(output_dir, "mcts", color_value)
            for local_index, seed in enumerate(subrun_seeds):
                game_output_dir = subrun_dir / "games" / f"game_{local_index:02d}_seed_{seed}"
                command = base.build_crossplay_command(
                    args.python,
                    job,
                    [seed],
                    game_output_dir,
                    color_value,
                )
                tasks.append(
                    GameTaskSpec(
                        job_key=job.key,
                        job_index=job_index,
                        plan=args.plan,
                        subrun_name=subrun_name,
                        subrun_dir=subrun_dir,
                        game_index_within_subrun=local_index,
                        global_seed_index=local_index if color_value == "black" else len(black_seeds) + local_index,
                        seed=seed,
                        command=command,
                        output_dir=game_output_dir,
                        color_key=color_key,
                        color_value=color_value,
                    )
                )
    else:
        subruns = [
            ("standard_black", "standard_color", "black", black_seeds),
            ("standard_white", "standard_color", "white", white_seeds),
        ]
        for subrun_name, color_key, color_value, subrun_seeds in subruns:
            subrun_dir = base.build_subrun_output_dir(output_dir, "standard", color_value)
            for local_index, seed in enumerate(subrun_seeds):
                game_output_dir = subrun_dir / "games" / f"game_{local_index:02d}_seed_{seed}"
                command = base.build_ablation_command(
                    args.python,
                    job,
                    [seed],
                    game_output_dir,
                    color_value,
                )
                tasks.append(
                    GameTaskSpec(
                        job_key=job.key,
                        job_index=job_index,
                        plan=args.plan,
                        subrun_name=subrun_name,
                        subrun_dir=subrun_dir,
                        game_index_within_subrun=local_index,
                        global_seed_index=local_index if color_value == "black" else len(black_seeds) + local_index,
                        seed=seed,
                        command=command,
                        output_dir=game_output_dir,
                        color_key=color_key,
                        color_value=color_value,
                    )
                )

    return tasks


def run_game_task(task: GameTaskSpec) -> dict:
    result = base.run_job(task.command, str(base.REPO_ROOT))
    return {
        "task": task,
        **result,
    }


def task_result_for_manifest(result: dict) -> dict:
    task: GameTaskSpec = result["task"]
    return {
        "task": asdict(task),
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "duration_s": result["duration_s"],
    }


def aggregate_completed_job(job_state: dict) -> None:
    subrun_tasks: dict[str, list[GameTaskSpec]] = {}
    for task in job_state["tasks"]:
        subrun_tasks.setdefault(task.subrun_name, []).append(task)

    for subrun_name, tasks in subrun_tasks.items():
        aggregate_single_game_outputs(
            subrun_dir=tasks[0].subrun_dir,
            game_tasks=tasks,
        )

    if job_state["plan"] == "crossplay":
        base.merge_crossplay_job_outputs(
            job_state["job"],
            job_state["output_dir"],
            job_state["seeds"],
        )
    else:
        base.merge_ablation_job_outputs(
            job_state["job"],
            job_state["output_dir"],
            job_state["seeds"],
        )


def default_high_parallel_workers(task_count: int) -> int:
    return max(1, min(task_count, (os.cpu_count() or 2)))


def main() -> None:
    args = base.parse_args()
    jobs = base.plan_jobs(args.plan)
    groups = base.plan_groups(args.plan)
    output_root = base.ensure_output_root(args.output_root, args.plan)
    output_root.mkdir(parents=True, exist_ok=True)

    selected_job_keys = set(args.jobs) if args.jobs is not None else None
    scheduled_jobs = []
    job_index_rows = []

    for job_index, job in enumerate(jobs):
        if selected_job_keys is not None and job.key not in selected_job_keys:
            continue

        seeds = base.build_seed_list(job_index, job, args.base_seed)
        output_dir = base.build_job_output_dir(output_root, job)
        skip = args.skip_existing and base.should_skip_job(output_dir)

        if args.plan == "crossplay":
            primary_label = f"MCTS(rounds={job.mcts_rounds})"
            secondary_label = f"Minimax(depth={job.minimax_depth})"
        else:
            primary_label = job.variant_name
            secondary_label = job.baseline_name

        tasks = [] if skip else build_job_game_tasks(args, job, job_index, output_dir, seeds)
        scheduled_jobs.append(
            {
                "job": job,
                "plan": args.plan,
                "seeds": seeds,
                "output_dir": output_dir,
                "skip": skip,
                "tasks": tasks,
                "primary_label": primary_label,
                "secondary_label": secondary_label,
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
            for task in item["tasks"]:
                print(
                    f"  game[{task.subrun_name} #{task.game_index_within_subrun}] "
                    f"({task.color_key}={task.color_value}, seed={task.seed}):",
                    " ".join(task.command),
                )
        return

    max_workers = args.max_workers or default_high_parallel_workers(
        sum(len(item["tasks"]) for item in scheduled_jobs)
    )
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

    job_states: dict[str, dict] = {}
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

        job_states[job.key] = {
            "job": job,
            "plan": item["plan"],
            "seeds": item["seeds"],
            "output_dir": item["output_dir"],
            "tasks": item["tasks"],
            "pending": len(item["tasks"]),
            "start_time": time.perf_counter(),
            "failures": [],
            "task_results": [],
        }

    future_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for state in job_states.values():
            for task in state["tasks"]:
                future = executor.submit(run_game_task, task)
                future_map[future] = task

        for future in as_completed(future_map):
            result = future.result()
            task: GameTaskSpec = result["task"]
            state = job_states[task.job_key]
            state["task_results"].append(result)
            state["pending"] -= 1
            if result["returncode"] != 0:
                state["failures"].append(result)

            if state["pending"] != 0:
                continue

            duration_s = time.perf_counter() - state["start_time"]
            if state["failures"]:
                status = "failed"
            else:
                aggregate_completed_job(state)
                status = "completed"

            manifest["jobs"].append(
                {
                    "plan": state["plan"],
                    "job": asdict(state["job"]),
                    "seeds": state["seeds"],
                    "output_dir": str(state["output_dir"]),
                    "status": status,
                    "duration_s": duration_s,
                    "returncode": 0 if status == "completed" else 1,
                    "subruns": [task_result_for_manifest(item) for item in state["task_results"]],
                }
            )

            for row in job_index_rows:
                if row["job_key"] != task.job_key:
                    continue
                row["status"] = status
                row["duration_s"] = f"{duration_s:.3f}"
                break

            print(f"[{status}] {task.job_key} ({duration_s:.2f}s)")
            if status == "failed":
                for failed in state["failures"]:
                    if failed["stderr"]:
                        print(failed["stderr"])

    base.write_job_index(output_root / "job_index.csv", job_index_rows)
    base.write_manifest(output_root / "manifest.json", manifest)

    failed_jobs = [job for job in manifest["jobs"] if job["status"] == "failed"]
    print(f"Saved job index to {output_root / 'job_index.csv'}")
    print(f"Saved manifest to {output_root / 'manifest.json'}")
    if failed_jobs:
        raise SystemExit(f"{len(failed_jobs)} job(s) failed.")


if __name__ == "__main__":
    main()
