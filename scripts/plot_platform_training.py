"""Plot learning curves from platform DQN metrics.csv.

The script keeps the old single-run workflow, and also understands the
platform-specific metrics added for the two-sided requester/worker setting.
It derives `reward` from `platform_reward` for old reward plots, but keeps
worker/requester hit-rate, epsilon, and loss as separate two-sided signals.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TEXT_COLUMNS = {"split", "time", "run"}
VALIDATION_SCORE_COLUMNS = {
    "avg_worker_utility",
    "avg_requester_utility",
    "worker_hit_rate",
    "requester_hit_rate",
    "avg_requester_pool_size",
    "requester_decisions",
    "avg_project_wait_days",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot platform training curves")
    parser.add_argument(
        "metrics",
        type=Path,
        nargs="+",
        help="One or more metrics.csv paths. Multiple files produce comparison plots.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional run labels matching the metrics paths.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Moving average window for plotted curves; 1 disables smoothing.",
    )
    parser.add_argument(
        "--compare-split",
        choices=["train", "val"],
        default="val",
        help="Split used when comparing multiple runs.",
    )
    args = parser.parse_args()

    labels = build_labels(args.metrics, args.labels)
    out_dir = args.out_dir or args.metrics[0].parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        load_metrics(metrics_path, label)
        for metrics_path, label in zip(args.metrics, labels, strict=True)
    ]

    if len(frames) == 1:
        df = frames[0]
        plot_single_run(df, out_dir, args.smooth_window)
        df.to_csv(out_dir / "normalized_metrics.csv", index=False)
    else:
        combined = pd.concat(frames, ignore_index=True)
        plot_run_comparisons(
            combined,
            out_dir,
            split=args.compare_split,
            smooth_window=args.smooth_window,
        )
        combined.to_csv(out_dir / "normalized_metrics.csv", index=False)

    summary = summarize(pd.concat(frames, ignore_index=True))
    summary_path = out_dir / "training_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"plots: {out_dir}")
    print(f"normalized_metrics: {out_dir / 'normalized_metrics.csv'}")
    print(f"summary: {summary_path}")


def build_labels(metrics_paths: list[Path], labels: list[str] | None) -> list[str]:
    if labels is not None and len(labels) != len(metrics_paths):
        raise ValueError("--labels length must match the number of metrics paths")
    if labels:
        return labels
    if len(metrics_paths) == 1:
        return [metrics_paths[0].parent.name]
    return [path.parent.parent.name or path.parent.name for path in metrics_paths]


def load_metrics(metrics_path: Path, label: str) -> pd.DataFrame:
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_path}")

    df = pd.read_csv(metrics_path)
    for col in df.columns:
        if col not in TEXT_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["run"] = label
    add_compatibility_columns(df)
    add_validation_score(df)
    return df


def add_compatibility_columns(df: pd.DataFrame) -> None:
    """Add non-averaged compatibility columns when platform columns are available."""
    if "reward" not in df.columns:
        if "platform_reward" in df.columns:
            df["reward"] = df["platform_reward"]
        elif {"worker_reward", "requester_reward"} <= set(df.columns):
            df["reward"] = df["worker_reward"].fillna(0.0) + df[
                "requester_reward"
            ].fillna(0.0)


def add_validation_score(df: pd.DataFrame) -> None:
    if "validation_score" in df.columns:
        return
    if not VALIDATION_SCORE_COLUMNS <= set(df.columns):
        return
    df["validation_score"] = (
        df["avg_worker_utility"]
        + 5.0 * df["avg_requester_utility"]
        + 0.5 * df["worker_hit_rate"]
        + 2.0 * df["requester_hit_rate"]
        - 0.02 * df["avg_requester_pool_size"]
        - 0.001 * df["requester_decisions"]
        - 0.05 * df["avg_project_wait_days"]
    )


def plot_single_run(df: pd.DataFrame, out_dir: Path, smooth_window: int) -> None:
    plot_lines(
        df,
        ["reward", "platform_reward", "worker_reward", "requester_reward"],
        out_dir / "reward_curves.png",
        "Reward curves",
        "Reward",
        smooth_window,
    )
    plot_reward_trend(df, out_dir / "reward_trend_curves.png", smooth_window)
    plot_lines(
        df,
        ["hit_rate", "worker_hit_rate", "requester_hit_rate"],
        out_dir / "hit_rate_curves.png",
        "Hit-rate curves",
        "Hit rate",
        smooth_window,
    )
    plot_lines(
        df,
        ["validation_score"],
        out_dir / "validation_score_curves.png",
        "Validation score",
        "Score",
        smooth_window,
    )
    plot_lines(
        df,
        ["avg_worker_utility", "avg_requester_utility"],
        out_dir / "utility_curves.png",
        "Utility curves",
        "Utility",
        smooth_window,
    )
    plot_lines(
        df,
        [
            "requester_wait_rate",
            "requester_waited_project_rate",
            "avg_requester_waits_per_project",
        ],
        out_dir / "requester_wait_curves.png",
        "Requester WAIT behavior",
        "Value",
        smooth_window,
    )
    plot_lines(
        df,
        ["avg_requester_pool_size"],
        out_dir / "requester_pool_size_curves.png",
        "Requester candidate pool size",
        "Pool size",
        smooth_window,
    )
    plot_lines(
        df,
        [
            "project_wait_cost",
            "avg_project_wait_days",
            "rerouted_workers",
            "requester_waits",
        ],
        out_dir / "dynamic_state_curves.png",
        "Dynamic platform state",
        "Value",
        smooth_window,
    )
    plot_lines(
        df,
        ["avg_loss", "worker_avg_loss", "requester_avg_loss"],
        out_dir / "loss_curves.png",
        "Training loss",
        "Loss",
        smooth_window,
    )
    plot_lines(
        df,
        ["epsilon", "worker_epsilon", "requester_epsilon"],
        out_dir / "epsilon_curves.png",
        "Epsilon schedule",
        "Epsilon",
        smooth_window,
    )
    plot_focus_dashboard(df, out_dir / "platform_focus_curves.png", smooth_window)


def plot_lines(
    df: pd.DataFrame,
    columns: list[str],
    output: Path,
    title: str,
    ylabel: str,
    smooth_window: int,
) -> None:
    available = [col for col in columns if col in df.columns]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for split, split_df in df.groupby("split", sort=False):
        split_df = split_df.sort_values("episode")
        for col in available:
            series = split_df[col]
            if not series.notna().any():
                continue
            ax.plot(
                split_df["episode"],
                smooth_series(series, smooth_window),
                marker="o",
                linewidth=1.8,
                label=f"{split}_{col}",
            )
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_reward_trend(df: pd.DataFrame, output: Path, smooth_window: int) -> None:
    reward_col = first_available(df, ["reward", "platform_reward"])
    if reward_col is None:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for split, split_df in df.groupby("split", sort=False):
        split_df = split_df.sort_values("episode").copy()
        reward = split_df[reward_col]
        if not reward.notna().any():
            continue
        ax.plot(
            split_df["episode"],
            reward.cummax(),
            marker="o",
            linewidth=2.2,
            label=f"{split}_running_best",
        )
        ax.plot(
            split_df["episode"],
            smooth_series(reward, max(3, smooth_window)),
            marker=".",
            linewidth=1.5,
            linestyle="--",
            label=f"{split}_ma",
        )
    ax.set_title("Reward trend")
    ax.set_xlabel("Episode")
    ax.set_ylabel(reward_col)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_focus_dashboard(df: pd.DataFrame, output: Path, smooth_window: int) -> None:
    specs = [
        ("validation_score", "Validation score", "Score"),
        ("avg_requester_utility", "Requester utility", "Utility"),
        ("requester_hit_rate", "Requester hit", "Hit rate"),
        ("requester_wait_rate", "Requester WAIT rate", "WAIT rate"),
        ("avg_requester_pool_size", "Requester pool size", "Pool size"),
        ("platform_reward_per_step", "Reward per step", "Reward/step"),
    ]
    specs = [spec for spec in specs if spec[0] in df.columns]
    if not specs:
        return

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    axes_flat = axes.ravel()
    for ax, (metric, title, ylabel) in zip(axes_flat, specs):
        for split, split_df in df.groupby("split", sort=False):
            split_df = split_df.sort_values("episode")
            series = split_df[metric]
            if not series.notna().any():
                continue
            ax.plot(
                split_df["episode"],
                smooth_series(series, smooth_window),
                linewidth=1.8,
                label=split,
            )
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    for ax in axes_flat[len(specs) :]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncols=2, frameon=False)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_run_comparisons(
    df: pd.DataFrame,
    out_dir: Path,
    split: str,
    smooth_window: int,
) -> None:
    metrics = [
        ("validation_score", "Validation score", "Score"),
        ("reward", "Reward", "Reward"),
        ("hit_rate", "Hit rate", "Hit rate"),
        ("worker_hit_rate", "Worker hit", "Hit rate"),
        ("requester_hit_rate", "Requester hit", "Hit rate"),
        ("avg_requester_utility", "Requester utility", "Utility"),
        ("requester_wait_rate", "Requester WAIT rate", "WAIT rate"),
        ("avg_requester_pool_size", "Requester pool size", "Pool size"),
        ("epsilon", "Epsilon", "Epsilon"),
        ("worker_epsilon", "Worker epsilon", "Epsilon"),
        ("requester_epsilon", "Requester epsilon", "Epsilon"),
        ("avg_loss", "Loss", "Loss"),
        ("worker_avg_loss", "Worker loss", "Loss"),
        ("requester_avg_loss", "Requester loss", "Loss"),
    ]
    for metric, title, ylabel in metrics:
        if metric not in df.columns:
            continue
        plot_compare_metric(
            df,
            metric,
            out_dir / f"compare_{metric}.png",
            title,
            ylabel,
            split,
            smooth_window,
        )

    dashboard_metrics = [
        ("validation_score", "Validation score", "Score"),
        ("avg_requester_utility", "Requester utility", "Utility"),
        ("requester_hit_rate", "Requester hit", "Hit rate"),
        ("requester_wait_rate", "Requester WAIT rate", "WAIT rate"),
        ("avg_requester_pool_size", "Requester pool size", "Pool size"),
        ("reward", "Reward", "Reward"),
    ]
    plot_compare_dashboard(
        df,
        [spec for spec in dashboard_metrics if spec[0] in df.columns],
        out_dir / "compare_platform_focus.png",
        split,
        smooth_window,
    )


def plot_compare_metric(
    df: pd.DataFrame,
    metric: str,
    output: Path,
    title: str,
    ylabel: str,
    split: str,
    smooth_window: int,
) -> None:
    split_df = df[df["split"] == split].copy()
    if split_df.empty or not split_df[metric].notna().any():
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for run, run_df in split_df.groupby("run", sort=False):
        run_df = run_df.sort_values("episode")
        series = run_df[metric]
        if not series.notna().any():
            continue
        ax.plot(
            run_df["episode"],
            smooth_series(series, smooth_window),
            linewidth=1.9,
            label=run,
        )
    ax.set_title(f"{title} ({split})")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_compare_dashboard(
    df: pd.DataFrame,
    specs: list[tuple[str, str, str]],
    output: Path,
    split: str,
    smooth_window: int,
) -> None:
    if not specs:
        return
    split_df = df[df["split"] == split].copy()
    if split_df.empty:
        return

    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    axes_flat = axes.ravel()
    for ax, (metric, title, ylabel) in zip(axes_flat, specs):
        for run, run_df in split_df.groupby("run", sort=False):
            run_df = run_df.sort_values("episode")
            series = run_df[metric]
            if not series.notna().any():
                continue
            ax.plot(
                run_df["episode"],
                smooth_series(series, smooth_window),
                linewidth=1.8,
                label=run,
            )
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    for ax in axes_flat[len(specs) :]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncols=3, frameon=False)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (run, split), split_df in df.groupby(["run", "split"], sort=False):
        split_df = split_df.sort_values("episode")
        score_col = first_available(
            split_df,
            ["validation_score", "platform_reward", "reward"],
        )
        if score_col is None:
            continue
        best_idx = split_df[score_col].idxmax()
        best = split_df.loc[best_idx]
        final = split_df.iloc[-1]
        row = {
            "run": run,
            "split": split,
            "score_column": score_col,
            "best_episode": int(best["episode"]),
            "best_score": best[score_col],
            "final_episode": int(final["episode"]),
            "final_score": final[score_col],
        }
        for prefix, source in (("best", best), ("final", final)):
            for col in [
                "validation_score",
                "reward",
                "hit_rate",
                "epsilon",
                "avg_worker_utility",
                "avg_requester_utility",
                "worker_hit_rate",
                "requester_hit_rate",
                "avg_loss",
                "worker_avg_loss",
                "requester_avg_loss",
                "worker_epsilon",
                "requester_epsilon",
                "requester_wait_rate",
                "avg_requester_pool_size",
                "platform_reward_per_step",
                "project_wait_cost",
                "rerouted_workers",
            ]:
                if col in split_df.columns:
                    row[f"{prefix}_{col}"] = source[col]
        rows.append(row)
    return pd.DataFrame(rows)


def first_available(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            return col
    return None


if __name__ == "__main__":
    main()
