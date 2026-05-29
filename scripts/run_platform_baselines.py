"""Run heuristic baselines for the dynamic platform simulation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.platform_env import PlatformEnvConfig, PlatformSimulationEnv
from models.platform_baselines import make_platform_selectors
from models.platform_training import run_platform_eval
from scripts.evaluate_platform import build_with_limit, load_platform_agent
from src.platform_dataset import PlatformDataset


DEFAULT_POLICIES = [
    ("random_project", "wait_until_deadline"),
    ("popularity", "worker_quality"),
    ("category_match", "worker_category_match"),
    ("industry_match", "worker_industry_match"),
    ("award", "worker_quality"),
    ("low_wait_project", "worker_quality"),
    ("joint_heuristic", "worker_quality"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dynamic platform baselines")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-projects", type=int, default=0)
    parser.add_argument("--num-project-candidates", type=int, default=32)
    parser.add_argument("--num-worker-candidates", type=int, default=32)
    parser.add_argument("--include-truth-in-candidates", action="store_true")
    parser.add_argument("--project-wait-penalty", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--worker-checkpoint", type=str, default=None)
    parser.add_argument("--requester-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", default="runs/platform_baselines")
    args = parser.parse_args()

    ds = build_with_limit(args.max_projects)
    rows: list[dict] = []
    for worker_policy, requester_policy in DEFAULT_POLICIES:
        print(f"评估 {worker_policy} + {requester_policy} ...", flush=True)
        metrics = evaluate_pair(args, ds, worker_policy, requester_policy)
        row = {
            "policy": f"{worker_policy}+{requester_policy}",
            "worker_policy": worker_policy,
            "requester_policy": requester_policy,
            **metrics,
            "include_truth_in_candidates": args.include_truth_in_candidates,
        }
        rows.append(row)
        print(
            f"  platform_reward={row['platform_reward']:.2f} "
            f"worker_hit={row['worker_hit_rate']:.4f} "
            f"requester_hit={row['requester_hit_rate']:.4f}",
            flush=True,
        )

    if args.worker_checkpoint and args.requester_checkpoint:
        print("评估 dqn+dqn ...", flush=True)
        metrics = evaluate_dqn(args, ds)
        rows.append(
            {
                "policy": "dqn+dqn",
                "worker_policy": "dqn",
                "requester_policy": "dqn",
                **metrics,
                "include_truth_in_candidates": args.include_truth_in_candidates,
            }
        )

    out_dir = Path(args.output_dir) / platform_dir_name(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "comparison.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    csv_path = out_dir / "comparison.csv"
    fields = [
        "policy",
        "worker_policy",
        "requester_policy",
        "worker_hit_rate",
        "requester_hit_rate",
        "worker_reward",
        "requester_reward",
        "platform_reward",
        "project_wait_cost",
        "avg_project_wait_days",
        "filled_project_rate",
        "winner_quality",
        "rerouted_workers",
        "closed_projects",
        "unfilled_projects",
        "steps",
        "include_truth_in_candidates",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    best = max(rows, key=lambda row: row["platform_reward"])
    print(f"\n对比表: {csv_path}", flush=True)
    print(f"完整 JSON: {json_path}", flush=True)
    print(
        f"最佳 Platform Reward: {best['policy']} -> {best['platform_reward']:.2f}",
        flush=True,
    )


def evaluate_pair(args, ds, worker_policy: str, requester_policy: str) -> dict:
    env = build_env(args, ds)
    worker_select, requester_select = make_platform_selectors(
        worker_policy,
        requester_policy,
    )
    return run_platform_eval(
        env,
        worker_select,
        requester_select,
        max_steps=None if args.max_steps == 0 else args.max_steps,
    )


def evaluate_dqn(args, ds) -> dict:
    env = build_env(args, ds)
    worker_agent = load_platform_agent(
        Path(args.worker_checkpoint),
        args.num_project_candidates,
        "worker",
    )
    requester_agent = load_platform_agent(
        Path(args.requester_checkpoint),
        args.num_worker_candidates + 1,
        "requester",
    )

    def worker_select(_env, decision):
        return worker_agent.select_action(decision.observation, explore=False)

    def requester_select(_env, decision):
        return requester_agent.select_action(decision.observation, explore=False)

    return run_platform_eval(
        env,
        worker_select,
        requester_select,
        max_steps=None if args.max_steps == 0 else args.max_steps,
    )


def build_env(args, ds) -> PlatformSimulationEnv:
    platform = PlatformDataset(ds, args.split)
    cfg = PlatformEnvConfig(
        num_project_candidates=args.num_project_candidates,
        num_worker_candidates=args.num_worker_candidates,
        include_truth_in_candidates=args.include_truth_in_candidates,
        project_wait_penalty=args.project_wait_penalty,
    )
    return PlatformSimulationEnv(platform, cfg, seed=42)


def platform_dir_name(args) -> str:
    truth = "with_truth" if args.include_truth_in_candidates else "no_truth"
    return f"platform_{args.split}_{truth}"


if __name__ == "__main__":
    main()
