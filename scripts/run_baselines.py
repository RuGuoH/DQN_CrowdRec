"""
批量运行基线并在 test 集上对比（可附加 DQN checkpoint）。

示例:
  python scripts/run_baselines.py --side worker --max-projects 100
  python scripts/run_baselines.py --side worker --checkpoint runs/worker/.../best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.baselines import REQUESTER_BASELINES, WORKER_BASELINES
from models.eval_runner import evaluate_one
from src.config import Config, load_config
from src.dataset import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="批量评估基线与 DQN")
    parser.add_argument("--side", choices=["worker", "requester"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-projects", type=int, default=0)
    parser.add_argument("--num-candidates", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None, help="可选 DQN best.pt")
    parser.add_argument("--output-dir", type=str, default="runs/baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
    "--no-truth-in-candidates",
    action="store_true",
    help="Do not force ground-truth item into candidate set.",
)
    args = parser.parse_args()

    cfg = load_config()
    max_p = None if args.max_projects == 0 else args.max_projects
    if max_p is not None:
        cfg = Config(
            cfg.data_dir, cfg.min_start_date, cfg.page_limit,
            cfg.train_ratio, cfg.val_ratio, max_p, cfg.cache_dir,
        )
    ds = build_dataset(cfg)

    names = list(
        (WORKER_BASELINES if args.side == "worker" else REQUESTER_BASELINES).keys()
    )
    if args.checkpoint:
        names.append("dqn")

    results: list[dict] = []
    for name in names:
        policy = "dqn" if name == "dqn" else name
        ckpt = Path(args.checkpoint) if name == "dqn" else None
        print(f"评估 {policy} ...", flush=True)
        r = evaluate_one(
            args.side,
            ds,
            args.split,  # type: ignore[arg-type]
            policy=policy,
            checkpoint=ckpt,
            num_candidates=args.num_candidates,
            max_steps=None if args.max_steps == 0 else args.max_steps,
            seed=args.seed,
            include_truth_in_candidates=not args.no_truth_in_candidates,
        )
        results.append(r)
        print(
            f"  hit={r['hit_rate']:.4f} reward={r['reward']:.2f} steps={r['steps']}",
            flush=True,
        )
    truth_tag = (
    "no_truth"
    if args.no_truth_in_candidates
    else "with_truth"
)
    out_dir = Path(args.output_dir) / f"{args.side}_{args.split}_{truth_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "comparison.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    csv_path = out_dir / "comparison.csv"
    fields = ["policy", "hit_rate", "reward", "steps", "hits", "num_events","include_truth_in_candidates"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fields}
            row["policy"] = r.get("policy", "")
            w.writerow(row)

    print(f"\n对比表: {csv_path}", flush=True)
    print(f"完整 JSON: {json_path}", flush=True)

    best = max(results, key=lambda x: x["hit_rate"])
    print(f"最佳 Hit@1: {best['policy']} -> {best['hit_rate']:.4f}", flush=True)


if __name__ == "__main__":
    main()
