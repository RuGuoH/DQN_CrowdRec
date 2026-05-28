"""Plot learning curves from platform DQN metrics.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot platform training curves")
    parser.add_argument("metrics", type=Path, help="Path to metrics.csv")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    metrics_path = args.metrics
    out_dir = args.out_dir or metrics_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_path)
    for col in df.columns:
        if col not in {"split", "time"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    plot_lines(
        df,
        ["platform_reward", "worker_reward", "requester_reward"],
        out_dir / "reward_curves.png",
        "Reward curves",
        "Reward",
    )
    plot_reward_trend(
        df,
        out_dir / "reward_trend_curves.png",
    )
    plot_lines(
        df,
        ["worker_hit_rate", "requester_hit_rate"],
        out_dir / "hit_rate_curves.png",
        "Hit-rate curves",
        "Hit rate",
    )
    plot_lines(
        df,
        ["project_wait_cost", "avg_project_wait_days", "rerouted_workers"],
        out_dir / "dynamic_state_curves.png",
        "Dynamic platform state",
        "Value",
    )
    plot_lines(
        df,
        ["worker_avg_loss", "requester_avg_loss"],
        out_dir / "loss_curves.png",
        "Training loss",
        "Loss",
    )
    plot_lines(
        df,
        ["worker_epsilon", "requester_epsilon"],
        out_dir / "epsilon_curves.png",
        "Epsilon schedule",
        "Epsilon",
    )

    summary = summarize(df)
    summary_path = out_dir / "training_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"plots: {out_dir}")
    print(f"summary: {summary_path}")


def plot_lines(
    df: pd.DataFrame,
    columns: list[str],
    output: Path,
    title: str,
    ylabel: str,
) -> None:
    available = [col for col in columns if col in df.columns]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for split, split_df in df.groupby("split", sort=False):
        split_df = split_df.sort_values("episode")
        for col in available:
            if split_df[col].notna().any():
                ax.plot(
                    split_df["episode"],
                    split_df[col],
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


def plot_reward_trend(df: pd.DataFrame, output: Path) -> None:
    if "platform_reward" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for split, split_df in df.groupby("split", sort=False):
        split_df = split_df.sort_values("episode").copy()
        reward = split_df["platform_reward"]
        if not reward.notna().any():
            continue
        split_df["platform_reward_running_best"] = reward.cummax()
        split_df["platform_reward_ma3"] = reward.rolling(
            window=3,
            min_periods=1,
        ).mean()
        ax.plot(
            split_df["episode"],
            split_df["platform_reward_running_best"],
            marker="o",
            linewidth=2.2,
            label=f"{split}_running_best",
        )
        ax.plot(
            split_df["episode"],
            split_df["platform_reward_ma3"],
            marker=".",
            linewidth=1.5,
            linestyle="--",
            label=f"{split}_ma3",
        )
    ax.set_title("Platform reward trend")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Platform reward")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for split, split_df in df.groupby("split", sort=False):
        best_idx = split_df["platform_reward"].idxmax()
        best = split_df.loc[best_idx]
        final = split_df.sort_values("episode").iloc[-1]
        rows.append(
            {
                "split": split,
                "best_episode": int(best["episode"]),
                "best_platform_reward": best["platform_reward"],
                "best_worker_hit_rate": best["worker_hit_rate"],
                "best_requester_hit_rate": best["requester_hit_rate"],
                "final_episode": int(final["episode"]),
                "final_platform_reward": final["platform_reward"],
                "final_worker_hit_rate": final["worker_hit_rate"],
                "final_requester_hit_rate": final["requester_hit_rate"],
                "final_project_wait_cost": final["project_wait_cost"],
                "final_rerouted_workers": final["rerouted_workers"],
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
