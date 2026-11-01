"""Plot measured training/evaluation CSV files without extrapolating results.

Figure contract: show whether observed collective returns change during IPPO
training and how the saved policy compares with random actions on held-out
episodes. Each figure is one quantitative panel. The training curve represents
one run, not repeated training seeds. All observations are retained; a trailing
mean is explicitly labeled. Export editable PDF/SVG and a 600-dpi PNG.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No recorded episodes in {path}")
    return rows


def trailing_mean(values, window):
    values = np.asarray(values, dtype=np.float64)
    if window < 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("A positive window and finite, non-empty data are required")
    prefix = np.r_[0.0, np.cumsum(values)]
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(0, ends - window)
    return (prefix[ends] - prefix[starts]) / (ends - starts)


def configure_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 11,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "legend.frameon": False,
    })


def export_figure(fig, stem, notes):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError:
        if len(fig.axes) != 1:
            raise RuntimeError("Multi-panel exports require the panel-alignment auditor")
        alignment = {"verdict": "NOT APPLICABLE", "reason": "Single quantitative panel",
                     "plot_area_bounds_figure_fraction": list(fig.axes[0].get_position().bounds)}
        stem.with_suffix(".alignment.json").write_text(json.dumps(alignment, indent=2) + "\n")
    else:
        require_matplotlib_panel_alignment(fig, json_out=str(stem.with_suffix(".alignment.json")),
                                            strict=True, tolerance_pt=1.5, gutter_tolerance_pt=1.5)
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    notes["export_contract"] = "Coursework report: editable PDF/SVG and 600-dpi PNG; no journal TIFF submission is required."
    stem.with_suffix(".notes.json").write_text(json.dumps(notes, indent=2) + "\n")
    plt.close(fig)


def plot_learning_curve(csv_path, output, window=1000, target_episodes=100_000):
    rows = load_csv(csv_path)
    episodes = np.array([int(row["episode"]) for row in rows])
    returns = np.array([float(row["collective_return"]) for row in rows])
    if not np.isfinite(returns).all() or np.any(np.diff(episodes) <= 0):
        raise ValueError("Episode indices must strictly increase and returns must be finite")
    if not np.array_equal(episodes, np.arange(1, len(episodes) + 1)):
        raise ValueError("Episode CSV must start at episode 1 with no missing episodes")
    smoothed = trailing_mean(returns, window)
    algorithm = "IPPO" if "agent_1_return" in rows[0] else "PPO"
    fig, ax = plt.subplots(figsize=(7.204724, 4.212598))  # 183 x 107 mm
    fig.subplots_adjust(left=0.12, right=0.97, top=0.82, bottom=0.28)
    ax.plot(episodes, returns, color="#aab5c6", linewidth=0.35, alpha=0.55,
            rasterized=True, label="Every episode")
    ax.plot(episodes, smoothed, color="#28658b", linewidth=1.6,
            label=f"Trailing mean ({window:,} episodes)")
    ax.set_xlabel("Completed training episodes")
    ax.set_ylabel("Collective episodic return")
    ax.set_xlim(1, max(2, int(episodes[-1])))
    ax.set_ylim(bottom=min(0, float(returns.min())))
    ax.grid(axis="y", color="#e5e8ed", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.ticklabel_format(axis="x", style="plain")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, borderaxespad=0)
    fig.text(0.12, 0.94, f"{algorithm} learning on MultiGrid", fontsize=11, weight="bold")
    status = "Completed" if episodes[-1] >= target_episodes else "Partial run"
    fig.text(0.12, 0.115, f"{status}: {episodes[-1]:,} / {target_episodes:,} target episodes | one training run", fontsize=8)
    fig.text(0.12, 0.065, "Return = sum of all agents' rewards in one episode. Smoothing uses available preceding episodes.", fontsize=7)
    export_figure(fig, Path(output) / "learning_curve", {
        "source_data": str(Path(csv_path).resolve()), "input_rows": len(rows), "excluded_rows": 0,
        "measured_episodes": int(episodes[-1]), "requested_target_episodes": target_episodes,
        "target_reached": bool(episodes[-1] >= target_episodes), "training_runs": 1,
        "metric": "Sum of per-agent, undiscounted rewards over an episode",
        "smoothing": {"kind": "trailing arithmetic mean", "window": window,
                      "start_rule": "Use all available preceding episodes, including current episode"},
        "uncertainty": "No interval: one training run and temporally correlated episode returns",
        "extrapolation": "None; axis ends at last recorded episode", "archetype": "Single quantitative panel",
    })


def plot_evaluation(csv_path, output):
    rows = load_csv(csv_path)
    policies = ["random", "stochastic", "deterministic"]
    algorithm = "IPPO" if "agent_1_return" in rows[0] else "PPO"
    labels = ["Random", f"{algorithm} stochastic", f"{algorithm} deterministic"]
    colors = ["#969ca8", "#28658b", "#b37539"]
    fig, ax = plt.subplots(figsize=(7.204724, 4.212598))  # 183 x 107 mm
    fig.subplots_adjust(left=0.12, right=0.97, top=0.78, bottom=0.24)
    count = []
    for x, (policy, color) in enumerate(zip(policies, colors)):
        values = np.array([float(row["collective_return"]) for row in rows if row["policy"] == policy])
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"Evaluation rows for {policy} are missing or non-finite")
        count.append(len(values))
        # Deterministic horizontal layout only; measured returns are unchanged.
        offsets = 0.19 * np.sin(np.arange(len(values)) * 2.39996323)
        ax.scatter(x + offsets, values, s=8, color=color, alpha=0.35, linewidths=0, rasterized=True)
        mean = values.mean()
        sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
        ax.errorbar(x, mean, yerr=1.96 * sem, fmt="D", color="#17233d", markersize=5,
                    capsize=4, linewidth=1.4, zorder=3)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("Collective episodic return")
    ax.set_xlim(-0.6, 2.6)
    ax.margins(y=0.06)  # Keep zero-return observations and full intervals visible.
    ax.grid(axis="y", color="#e5e8ed", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.text(0.12, 0.94, "Held-out policy evaluation", fontsize=11, weight="bold")
    fig.text(0.12, 0.85, "Points: individual episodes | diamonds: mean | bars: approximate 95% confidence interval", fontsize=8)
    fig.text(0.12, 0.13, f"Episodes per policy: {count[0]}, {count[1]}, {count[2]} | matched initial-state seeds | one trained checkpoint", fontsize=8)
    fig.text(0.12, 0.075, "Intervals reflect evaluation episodes; they do not measure variation across independent training runs.", fontsize=7)
    export_figure(fig, Path(output) / "evaluation_comparison", {
        "source_data": str(Path(csv_path).resolve()), "input_rows": len(rows), "excluded_rows": 0,
        "policies": policies, "episodes_per_policy": count,
        "replicate_unit": "Held-out evaluation episode", "training_runs": 1,
        "center": "Arithmetic mean", "interval": "Mean +/- 1.96 times sample SD / sqrt(n)",
        "jitter": "Deterministic horizontal offsets used only to reduce overlap, not data modification",
        "test": "No significance test", "archetype": "Single quantitative panel",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--target-episodes", type=int, default=100_000)
    args = parser.parse_args()
    if args.window < 1 or args.target_episodes < 1:
        parser.error("window and target-episodes must be positive")
    configure_style()
    output = args.run_dir / "figures"
    plot_learning_curve(args.run_dir / "episodes.csv", output, args.window, args.target_episodes)
    evaluation = args.run_dir / "evaluation" / "episodes.csv"
    if evaluation.exists():
        plot_evaluation(evaluation, output)
    print(output.resolve())


if __name__ == "__main__":
    main()
