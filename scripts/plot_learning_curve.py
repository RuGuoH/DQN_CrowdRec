"""学习曲线绘制脚本。

用法示例:

# 单个 run
python scripts/plot_learning_curve.py \
    --run-dir runs/worker/worker_dqn_worker_dqn_no_truth_20260523_222938

# 多个 run 对比
python scripts/plot_learning_curve.py \
    --run-dir runs/worker/run1 \
    --compare-runs runs/worker/run2 runs/worker/run3

默认会生成:
- reward_curve.png
- hitrate_curve.png
- loss_curve.png
- epsilon_curve.png
- combined.png

支持:
- train / val 双曲线
- moving average smoothing
- 多 run 对比
- 自动保存到 plots/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11


METRIC_CONFIGS = {
    "reward": {
        "ylabel": "Reward",
        "title": "Training Reward over Episodes",
        "filename": "reward_curve.png",
    },
    "hit_rate": {
        "ylabel": "Hit@1",
        "title": "Hit@1 Performance over Episodes",
        "filename": "hitrate_curve.png",
    },
    "avg_loss": {
        "ylabel": "Average Loss",
        "title": "Loss Curve",
        "filename": "loss_curve.png",
    },
    "epsilon": {
        "ylabel": "Epsilon",
        "title": "Epsilon Decay Curve",
        "filename": "epsilon_curve.png",
    },
}


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    """简单 moving average smoothing。"""
    return series.rolling(window=window, min_periods=1).mean()



def load_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"未找到 metrics.csv: {metrics_path}")

    df = pd.read_csv(metrics_path)

    required_cols = {
        "episode",
        "split",
        "reward",
        "hit_rate",
        "epsilon",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"metrics.csv 缺少字段: {missing}")

    return df



def plot_single_metric(
    df: pd.DataFrame,
    metric: str,
    save_path: Path,
    smooth_window: int,
    run_label: str,
) -> None:
    cfg = METRIC_CONFIGS[metric]

    plt.figure(figsize=(7, 4.5))

    for split in sorted(df["split"].unique()):
        split_df = df[df["split"] == split].copy()

        if metric not in split_df.columns:
            continue

        split_df = split_df.dropna(subset=[metric])
        if len(split_df) == 0:
            continue

        x = split_df["episode"]
        y = smooth_series(split_df[metric], smooth_window)

        plt.plot(x, y, label=f"{run_label}-{split}")

    plt.xlabel("Episode")
    plt.ylabel(cfg["ylabel"])
    plt.title(cfg["title"])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()



def plot_combined(
    df: pd.DataFrame,
    save_path: Path,
    smooth_window: int,
    run_label: str,
) -> None:
    """绘制 reward + hit_rate 双图。"""

    fig, axes = plt.subplots(2, 1, figsize=(7, 8))

    metrics = ["reward", "hit_rate"]

    for ax, metric in zip(axes, metrics):
        cfg = METRIC_CONFIGS[metric]

        for split in sorted(df["split"].unique()):
            split_df = df[df["split"] == split].copy()
            split_df = split_df.dropna(subset=[metric])

            if len(split_df) == 0:
                continue

            x = split_df["episode"]
            y = smooth_series(split_df[metric], smooth_window)

            ax.plot(x, y, label=f"{run_label}-{split}")

        ax.set_xlabel("Episode")
        ax.set_ylabel(cfg["ylabel"])
        ax.set_title(cfg["title"])
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()



def plot_compare_runs(
    run_dirs: list[Path],
    metric: str,
    save_path: Path,
    smooth_window: int,
) -> None:
    """多个 run 对比（默认只比较 train 曲线）。"""

    cfg = METRIC_CONFIGS[metric]

    plt.figure(figsize=(7, 4.5))

    for run_dir in run_dirs:
        df = load_metrics(run_dir)

        train_df = df[df["split"] == "train"].copy()
        train_df = train_df.dropna(subset=[metric])

        if len(train_df) == 0:
            continue

        x = train_df["episode"]
        y = smooth_series(train_df[metric], smooth_window)

        plt.plot(x, y, label=run_dir.name)

    plt.xlabel("Episode")
    plt.ylabel(cfg["ylabel"])
    plt.title(f"{cfg['title']} (Run Comparison)")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()



def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 DQN 学习曲线")

    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="单个训练 run 目录",
    )

    parser.add_argument(
        "--compare-runs",
        nargs="*",
        default=None,
        help="额外对比的 run 目录",
    )

    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="moving average window",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认 run_dir/plots）",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else run_dir / "plots"
    )

    df = load_metrics(run_dir)

    print(f"读取 metrics: {run_dir / 'metrics.csv'}")
    print(f"输出目录: {output_dir}")

    for metric in METRIC_CONFIGS:
        save_path = output_dir / METRIC_CONFIGS[metric]["filename"]

        plot_single_metric(
            df=df,
            metric=metric,
            save_path=save_path,
            smooth_window=args.smooth_window,
            run_label=run_dir.name,
        )

        print(f"已生成: {save_path}")

    combined_path = output_dir / "combined.png"

    plot_combined(
        df=df,
        save_path=combined_path,
        smooth_window=args.smooth_window,
        run_label=run_dir.name,
    )

    print(f"已生成: {combined_path}")

    if args.compare_runs:
        compare_dirs = [run_dir] + [Path(p) for p in args.compare_runs]

        compare_output = output_dir / "comparisons"

        for metric in ["reward", "hit_rate", "avg_loss"]:
            save_path = compare_output / f"compare_{metric}.png"

            plot_compare_runs(
                run_dirs=compare_dirs,
                metric=metric,
                save_path=save_path,
                smooth_window=args.smooth_window,
            )

            print(f"已生成对比图: {save_path}")

    print("\n学习曲线绘制完成。")


if __name__ == "__main__":
    main()
