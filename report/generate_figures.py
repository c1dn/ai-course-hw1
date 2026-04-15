from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = Path(__file__).resolve().parent
FIGURE_DIR = REPORT_ROOT / "figures"
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "results"


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_crossplay_summary(job_key: str) -> dict:
    return load_json(EXPERIMENT_ROOT / "experiment_plan" / job_key / "summary.json")


def load_ablation_summary(job_key: str) -> dict:
    return load_json(EXPERIMENT_ROOT / "mcts_ablation_plan" / job_key / "summary.json")


def parse_termination_summary(text: str) -> Counter:
    counter: Counter[str] = Counter()
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        name, count = part.split(":")
        counter[name.strip()] = int(count)
    return counter


def collect_round_sweep_rows() -> list[dict]:
    job_keys = [
        "mcts2200_vs_minimax5",
        "mcts3000_vs_minimax5",
        "mcts4000_vs_minimax5",
        "mcts5000_vs_minimax5",
    ]
    rows = []
    for job_key in job_keys:
        payload = load_crossplay_summary(job_key)
        summary = payload["summary"]
        rows.append(
            {
                "job_key": job_key,
                "rounds": summary["mcts_rounds"],
                "win_rate_percent": summary["mcts_win_rate_percent"],
                "avg_mcts_move_time_s": summary["avg_mcts_move_time_s"],
                "avg_total_duration_s": summary["avg_total_duration_s"],
                "max_mcts_move_time_s": summary["max_mcts_move_time_s"],
                "avg_mcts_margin": summary["avg_mcts_margin"],
                "mcts_exceeded_games": summary["mcts_exceeded_move_time_limit_games"],
                "terminations": parse_termination_summary(summary["termination_summary"]),
            }
        )
    return rows


def collect_round_color_rows() -> list[dict]:
    job_keys = [
        "mcts2200_vs_minimax5",
        "mcts3000_vs_minimax5",
        "mcts4000_vs_minimax5",
        "mcts5000_vs_minimax5",
    ]
    rows = []
    for job_key in job_keys:
        root = EXPERIMENT_ROOT / "experiment_plan" / job_key
        overall = load_crossplay_summary(job_key)["summary"]
        for subrun_name, color in [("mcts_black", "black"), ("mcts_white", "white")]:
            payload = load_json(root / subrun_name / "summary.json")
            summary = payload["summary"]
            mcts_win_rate_percent = (
                summary["black_win_rate_percent"]
                if color == "black"
                else summary["white_win_rate_percent"]
            )
            rows.append(
                {
                    "job_key": job_key,
                    "rounds": overall["mcts_rounds"],
                    "mcts_color": color,
                    "mcts_win_rate_percent": mcts_win_rate_percent,
                }
            )
    return rows


def collect_ablation_rows() -> list[dict]:
    job_keys = [
        "standard_vs_standard_1000",
        "low_c_vs_standard_1000",
        "cutoff10_eval_vs_standard_1000",
        "heuristic_rollout_vs_standard_1000",
        "heuristic_expansion_vs_standard_1000",
        "prior_bonus_vs_standard_1000",
        "full_optimized_vs_standard_1000",
    ]
    rows = []
    for job_key in job_keys:
        payload = load_ablation_summary(job_key)
        summary = payload["summary"]
        rows.append(
            {
                "job_key": job_key,
                "label": {
                    "standard_vs_standard_1000": "Standard vs Standard",
                    "low_c_vs_standard_1000": "Low c",
                    "cutoff10_eval_vs_standard_1000": "Cutoff-10 + eval",
                    "heuristic_rollout_vs_standard_1000": "Heuristic rollout",
                    "heuristic_expansion_vs_standard_1000": "Heuristic expansion",
                    "prior_bonus_vs_standard_1000": "Prior bonus",
                    "full_optimized_vs_standard_1000": "Full optimized",
                }[job_key],
                "variant_win_rate_percent": summary["variant_win_rate_percent"],
                "avg_variant_move_time_s": summary["avg_variant_move_time_s"],
                "avg_standard_move_time_s": summary["avg_standard_move_time_s"],
            }
        )
    return rows


def plot_round_sweep(rows: list[dict]) -> None:
    rounds = [row["rounds"] for row in rows]
    win_rates = [row["win_rate_percent"] for row in rows]
    move_times = [row["avg_mcts_move_time_s"] for row in rows]
    total_durations = [row["avg_total_duration_s"] for row in rows]
    move_limit_counts = [row["terminations"].get("move_limit_score", 0) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    ax = axes[0]
    bars = ax.bar(rounds, win_rates, width=430, color="#d07c3f", alpha=0.82)
    ax.set_title("MCTS vs Minimax(depth=5): strength vs cost")
    ax.set_xlabel("MCTS rounds")
    ax.set_ylabel("MCTS win rate (%)", color="#8a4f21")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelcolor="#8a4f21")
    for bar, value in zip(bars, win_rates):
        label_y = min(max(value + 2, 2), 98)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{value:.0f}%",
            ha="center",
            va="bottom" if value < 96 else "top",
            fontsize=9,
        )

    twin = ax.twinx()
    twin.plot(rounds, move_times, color="#1f5c84", marker="o", linewidth=2.2)
    twin.set_ylabel("Avg MCTS move time (s)", color="#1f5c84")
    twin.tick_params(axis="y", labelcolor="#1f5c84")

    ax = axes[1]
    ax.plot(rounds, total_durations, color="#2f7d4a", marker="o", linewidth=2.2)
    ax.set_title("Game duration and move-limit frequency")
    ax.set_xlabel("MCTS rounds")
    ax.set_ylabel("Avg total duration per game (s)", color="#2f7d4a")
    ax.tick_params(axis="y", labelcolor="#2f7d4a")

    twin = ax.twinx()
    twin.bar(rounds, move_limit_counts, width=430, color="#b25a7a", alpha=0.4)
    twin.set_ylabel("Move-limit games (out of 10)", color="#b25a7a")
    twin.set_ylim(0, 10)
    twin.tick_params(axis="y", labelcolor="#b25a7a")

    fig.savefig(FIGURE_DIR / "crossplay_round_sweep.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_color_split(rows: list[dict]) -> None:
    rounds = sorted({row["rounds"] for row in rows})
    black_rates = []
    white_rates = []
    for rounds_value in rounds:
        black_rates.append(
            next(
                row["mcts_win_rate_percent"]
                for row in rows
                if row["rounds"] == rounds_value and row["mcts_color"] == "black"
            )
        )
        white_rates.append(
            next(
                row["mcts_win_rate_percent"]
                for row in rows
                if row["rounds"] == rounds_value and row["mcts_color"] == "white"
            )
        )

    x = list(range(len(rounds)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    bars_black = ax.bar(
        [value - width / 2 for value in x],
        black_rates,
        width=width,
        color="#2e5f8f",
        label="MCTS as black",
    )
    bars_white = ax.bar(
        [value + width / 2 for value in x],
        white_rates,
        width=width,
        color="#c98d3d",
        label="MCTS as white",
    )

    ax.set_title("Strong color asymmetry at high MCTS budgets", fontsize=14)
    ax.set_xticks(x, [str(value) for value in rounds])
    ax.set_xlabel("MCTS rounds against Minimax(depth=5)")
    ax.set_ylabel("MCTS win rate (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncols=2)

    for collection in (bars_black, bars_white):
        for bar in collection:
            height = bar.get_height()
            label_y = min(max(height + 2, 2), 98)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{height:.0f}%",
                ha="center",
                va="bottom" if height < 96 else "top",
                fontsize=9,
            )

    fig.savefig(FIGURE_DIR / "crossplay_color_split.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_ablation(rows: list[dict]) -> None:
    labels = [row["label"] for row in rows]
    win_rates = [row["variant_win_rate_percent"] for row in rows]
    speedups = [
        row["avg_standard_move_time_s"] / row["avg_variant_move_time_s"]
        for row in rows
    ]
    y = list(range(len(labels)))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), constrained_layout=True)

    ax = axes[0]
    bars = ax.barh(y, win_rates, color="#b56d35", alpha=0.86)
    ax.set_title("Ablation result vs standard 1000-round MCTS")
    ax.set_xlabel("Variant win rate (%)")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    for bar, value in zip(bars, win_rates):
        ax.text(value + 1.5, bar.get_y() + bar.get_height() / 2, f"{value:.0f}%", va="center")

    ax = axes[1]
    bars = ax.barh(y, speedups, color="#3f7f63", alpha=0.86)
    ax.set_title("Speed relative to standard MCTS")
    ax.set_xlabel("Standard avg move time / Variant avg move time")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(speedups) * 1.2)
    ax.invert_yaxis()
    for bar, value in zip(bars, speedups):
        ax.text(value + 0.03, bar.get_y() + bar.get_height() / 2, f"{value:.2f}x", va="center")

    fig.savefig(FIGURE_DIR / "ablation_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    round_rows = collect_round_sweep_rows()
    plot_round_sweep(round_rows)
    plot_color_split(collect_round_color_rows())
    plot_ablation(collect_ablation_rows())
    print(f"Wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
