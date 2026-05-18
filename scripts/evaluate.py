"""
在指定数据划分上评估 DQN checkpoint 或基线策略。

示例:
  python scripts/evaluate.py --side worker --split test --policy random
  python scripts/evaluate.py --side worker --split test --checkpoint runs/worker/.../checkpoints/best.pt
"""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="评估 DQN 或基线策略")
    parser.add_argument("--side", choices=["worker", "requester"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--policy",
        default="dqn",
        help="dqn 或基线名",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-projects", type=int, default=0, help="0=全量")
    parser.add_argument("--num-candidates", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=0, help="0=不限制")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    baselines = WORKER_BASELINES if args.side == "worker" else REQUESTER_BASELINES
    if args.policy != "dqn" and args.policy not in baselines:
        raise SystemExit(f"未知 policy={args.policy}，可选: dqn, {list(baselines.keys())}")
    if args.policy == "dqn" and not args.checkpoint:
        raise SystemExit("policy=dqn 时必须提供 --checkpoint")

    cfg = load_config()
    max_p = None if args.max_projects == 0 else args.max_projects
    if max_p is not None:
        cfg = Config(
            cfg.data_dir, cfg.min_start_date, cfg.page_limit,
            cfg.train_ratio, cfg.val_ratio, max_p, cfg.cache_dir,
        )

    ds = build_dataset(cfg)
    result = evaluate_one(
        args.side,
        ds,
        args.split,  # type: ignore[arg-type]
        policy=args.policy,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        num_candidates=args.num_candidates,
        max_steps=None if args.max_steps == 0 else args.max_steps,
        seed=args.seed,
    )

    print(
        f"[{result['side']}/{result['split']}] policy={result['policy']} "
        f"hit={result['hit_rate']:.4f} reward={result['reward']:.2f} "
        f"steps={result['steps']} events={result['num_events']}",
        flush=True,
    )

    if args.output:
        out = Path(args.output)
    else:
        out_dir = ROOT / "runs" / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = result["policy"].replace(":", "_").replace("/", "_")
        out = out_dir / f"{args.side}_{args.split}_{safe}.json"

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果已写入: {out}", flush=True)


if __name__ == "__main__":
    main()
