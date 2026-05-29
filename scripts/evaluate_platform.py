"""Evaluate DQN or heuristic policies in the dynamic platform environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.platform_env import PlatformDecision, PlatformEnvConfig, PlatformSimulationEnv
from models.platform_baselines import make_platform_selectors
from models.platform_training import run_platform_eval
from src.config import Config, load_config
from src.dataset import build_dataset
from src.features import PROJECT_FEAT_DIM, WORKER_FEAT_DIM
from src.platform_dataset import PlatformDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dynamic platform policies")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-projects", type=int, default=0)
    parser.add_argument("--num-project-candidates", type=int, default=32)
    parser.add_argument("--num-worker-candidates", type=int, default=32)
    parser.add_argument("--worker-policy", default="dqn")
    parser.add_argument("--requester-policy", default="dqn")
    parser.add_argument("--worker-checkpoint", type=str, default=None)
    parser.add_argument("--requester-checkpoint", type=str, default=None)
    parser.add_argument("--include-truth-in-candidates", action="store_true")
    parser.add_argument("--project-wait-penalty", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    platform = PlatformDataset(build_with_limit(args.max_projects), args.split)
    env_cfg = PlatformEnvConfig(
        num_project_candidates=args.num_project_candidates,
        num_worker_candidates=args.num_worker_candidates,
        project_wait_penalty=args.project_wait_penalty,
        include_truth_in_candidates=args.include_truth_in_candidates,
    )
    env = PlatformSimulationEnv(platform, env_cfg, seed=42)

    worker_select, requester_select = build_selectors(args)
    metrics = run_platform_eval(
        env,
        worker_select,
        requester_select,
        max_steps=None if args.max_steps == 0 else args.max_steps,
    )
    result = {
        "split": args.split,
        "worker_policy": args.worker_policy,
        "requester_policy": args.requester_policy,
        "worker_checkpoint": args.worker_checkpoint,
        "requester_checkpoint": args.requester_checkpoint,
        "include_truth_in_candidates": args.include_truth_in_candidates,
        **metrics,
    }

    print(
        f"[platform/{args.split}] worker={args.worker_policy} "
        f"requester={args.requester_policy} "
        f"platform_reward={metrics['platform_reward']:.2f} "
        f"worker_hit={metrics['worker_hit_rate']:.4f} "
        f"requester_hit={metrics['requester_hit_rate']:.4f}",
        flush=True,
    )

    out = Path(args.output) if args.output else default_output(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果已写入: {out}", flush=True)


def build_selectors(args):
    if args.worker_policy == "dqn":
        if not args.worker_checkpoint:
            raise SystemExit("worker-policy=dqn requires --worker-checkpoint")
        worker_agent = load_platform_agent(
            Path(args.worker_checkpoint),
            args.num_project_candidates,
            "worker",
        )

        def worker_select(env: PlatformSimulationEnv, decision: PlatformDecision) -> int:
            return worker_agent.select_action(decision.observation, explore=False)

    else:
        worker_select, _ = make_platform_selectors(
            args.worker_policy,
            "worker_quality",
        )

    if args.requester_policy == "dqn":
        if not args.requester_checkpoint:
            raise SystemExit("requester-policy=dqn requires --requester-checkpoint")
        requester_agent = load_platform_agent(
            Path(args.requester_checkpoint),
            args.num_worker_candidates + 1,
            "requester",
        )

        def requester_select(
            env: PlatformSimulationEnv,
            decision: PlatformDecision,
        ) -> int:
            return requester_agent.select_action(decision.observation, explore=False)

    else:
        _, requester_select = make_platform_selectors(
            "random_project",
            args.requester_policy,
        )
    return worker_select, requester_select


def load_platform_agent(
    checkpoint: Path,
    num_actions: int,
    side: str,
) -> object:
    import torch
    from models.dqn import DQNAgent, DQNConfig

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    fields = DQNConfig.__dataclass_fields__
    cfg = DQNConfig(**{k: v for k, v in cfg_dict.items() if k in fields})
    if side == "worker":
        cfg.anchor_dim = WORKER_FEAT_DIM
        cfg.candidate_dim = PROJECT_FEAT_DIM
    else:
        cfg.anchor_dim = PROJECT_FEAT_DIM
        cfg.candidate_dim = WORKER_FEAT_DIM
    agent = DQNAgent(num_actions=num_actions, config=cfg)
    agent.load(checkpoint, load_optimizer=False)
    return agent


def build_with_limit(max_projects: int):
    cfg = load_config()
    max_p = None if max_projects == 0 else max_projects
    if max_p is not None:
        cfg = Config(
            cfg.data_dir,
            cfg.min_start_date,
            cfg.page_limit,
            cfg.train_ratio,
            cfg.val_ratio,
            max_p,
            cfg.cache_dir,
        )
    return build_dataset(cfg)


def default_output(args) -> Path:
    worker = args.worker_policy.replace("/", "_")
    requester = args.requester_policy.replace("/", "_")
    truth = "with_truth" if args.include_truth_in_candidates else "no_truth"
    return ROOT / "runs" / "platform_eval" / f"{args.split}_{worker}_{requester}_{truth}.json"


if __name__ == "__main__":
    main()
