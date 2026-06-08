"""Platform 动态双边环境的 BC 预训练。

示例:
  python scripts/pretrained_platform_bc.py --side worker --max-projects 50 --episodes 3
  python scripts/pretrained_platform_bc.py --side requester --max-projects 0 --episodes 5 --model dueling

  python scripts/train_platform_dqn.py \\
    --worker-pretrained runs/bc_platform/.../worker_best.pt \\
    --requester-pretrained runs/bc_platform/.../requester_best.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.platform_env import (
    PlatformEnvConfig,
    PlatformSimulationEnv,
    add_platform_env_cli_args,
    platform_env_config_from_args,
)
from models.dqn import DQNConfig, build_q_network
from models.training_log import EpisodeMetrics, TrainingLogger
from src.config import Config, load_config
from src.dataset import build_dataset
from src.features import (
    PLATFORM_PROJECT_FEAT_DIM,
    REQUESTER_CONTEXT_FEAT_DIM,
    WORKER_FEAT_DIM,
)
from src.platform_dataset import PlatformDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_with_limit(max_projects: int):
    cfg = load_config()
    max_p = None if max_projects == 0 else max_projects
    if max_p is not None:
        cfg = Config(
            data_dir=cfg.data_dir,
            min_start_date=cfg.min_start_date,
            page_limit=cfg.page_limit,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            max_projects=max_p,
            cache_dir=cfg.cache_dir,
        )
    return build_dataset(cfg)


@torch.no_grad()
def evaluate_platform_bc(
    env: PlatformSimulationEnv,
    model: torch.nn.Module,
    device: torch.device,
    side: str,
    *,
    max_steps: int | None = None,
) -> dict:
    decision = env.reset()
    hits = 0
    labels = 0
    steps = 0
    limit = max_steps or env.config.max_steps_per_episode

    while decision is not None:
        if decision.actor == side:
            obs = decision.observation
            anchor = torch.as_tensor(
                obs.worker_feat, dtype=torch.float32, device=device
            ).unsqueeze(0)
            cand = torch.as_tensor(
                obs.candidate_feat, dtype=torch.float32, device=device
            ).unsqueeze(0)
            mask = torch.as_tensor(obs.action_mask, device=device).unsqueeze(0)
            action = int(model(anchor, cand, mask).argmax(dim=1).item())

            if side == "worker":
                label = env.optimal_worker_action(decision.info.get("truth_project_id"))
            else:
                label = env.optimal_requester_action(int(decision.info["project_id"]))

            if label is not None:
                labels += 1
                hits += int(action == label)
        elif decision.actor == "worker":
            valid = np.flatnonzero(decision.observation.action_mask)
            action = int(valid[0]) if len(valid) else 0
        else:
            action = 0

        if decision.actor != side:
            step = env.step(action)
        else:
            step = env.step(action)

        steps += 1
        if step.terminated or (limit is not None and steps >= limit):
            break
        decision = step.decision

    metrics = env.final_metrics()
    metrics["bc_hit_rate"] = hits / max(labels, 1)
    metrics["bc_labeled_steps"] = float(labels)
    metrics["steps"] = float(steps)
    return metrics


def train_bc_episode(
    env: PlatformSimulationEnv,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    side: str,
    *,
    max_steps: int | None = None,
) -> dict:
    model.train()
    decision = env.reset()
    total_loss = 0.0
    updates = 0
    hits = 0
    labels = 0
    steps = 0
    limit = max_steps or env.config.max_steps_per_episode

    while decision is not None:
        if decision.actor == side:
            obs = decision.observation
            if side == "worker":
                label = env.optimal_worker_action(decision.info.get("truth_project_id"))
            else:
                label = env.optimal_requester_action(int(decision.info["project_id"]))

            if label is not None:
                anchor = torch.as_tensor(
                    obs.worker_feat, dtype=torch.float32, device=device
                ).unsqueeze(0)
                cand = torch.as_tensor(
                    obs.candidate_feat, dtype=torch.float32, device=device
                ).unsqueeze(0)
                mask = torch.as_tensor(obs.action_mask, device=device).unsqueeze(0)
                logits = model(anchor, cand, mask)
                target = torch.tensor([label], dtype=torch.long, device=device)
                loss = F.cross_entropy(logits, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                total_loss += float(loss.item())
                updates += 1
                pred = int(logits.argmax(dim=1).item())
                hits += int(pred == label)
                labels += 1
                action = label
            else:
                valid = np.flatnonzero(obs.action_mask)
                action = int(valid[0]) if len(valid) else 0

            step = env.step(action)
        elif decision.actor == "worker":
            valid = np.flatnonzero(decision.observation.action_mask)
            action = int(valid[0]) if len(valid) else 0
            step = env.step(action)
        else:
            step = env.step(0)

        steps += 1
        if step.terminated or (limit is not None and steps >= limit):
            break
        decision = step.decision

    return {
        "avg_loss": total_loss / max(updates, 1),
        "bc_hit_rate": hits / max(labels, 1),
        "bc_labeled_steps": float(labels),
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Platform BC pretraining")
    parser.add_argument("--side", choices=["worker", "requester"], required=True)
    parser.add_argument("--max-projects", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--num-project-candidates", type=int, default=32)
    parser.add_argument("--num-worker-candidates", type=int, default=32)
    parser.add_argument("--model", choices=["dqn", "dueling"], default="dueling")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64, help="unused, kept for CLI parity")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--extra-hidden-layers",
        type=int,
        default=1,
        help="在旧网络结构基础上额外增加的隐藏层数；需与 DQN 微调保持一致",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default="runs/bc_platform")
    parser.add_argument("--include-truth-in-candidates", action="store_true")
    add_platform_env_cli_args(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    ds = build_with_limit(args.max_projects)
    max_steps = None if args.max_steps == 0 else args.max_steps
    env_cfg = platform_env_config_from_args(
        args,
        max_steps_per_episode=max_steps,
    )

    train_env = PlatformSimulationEnv(PlatformDataset(ds, "train"), env_cfg, seed=args.seed)
    val_env = PlatformSimulationEnv(PlatformDataset(ds, "val"), env_cfg, seed=args.seed + 1)

    if args.side == "worker":
        num_actions = args.num_project_candidates
        anchor_dim = WORKER_FEAT_DIM
        candidate_dim = PLATFORM_PROJECT_FEAT_DIM
    else:
        num_actions = args.num_worker_candidates + 1
        anchor_dim = REQUESTER_CONTEXT_FEAT_DIM
        candidate_dim = WORKER_FEAT_DIM

    device = torch.device(args.device)
    model = build_q_network(
        model_type=args.model,
        num_actions=num_actions,
        hidden_dim=args.hidden_dim,
        anchor_dim=anchor_dim,
        candidate_dim=candidate_dim,
        extra_hidden_layers=args.extra_hidden_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    truth_tag = "with_truth" if args.include_truth_in_candidates else "no_truth"
    logger = TrainingLogger(
        Path(args.log_dir),
        run_name=f"bc_platform_{args.side}_{args.model}_{args.reward_mode}_{truth_tag}",
    )
    logger.save_config(
        {
            "side": args.side,
            "dataset": ds.summary(),
            "env": vars(env_cfg),
            "model": args.model,
            "episodes": args.episodes,
        }
    )
    print(f"日志目录: {logger.run_dir}", flush=True)

    best_val = float("-inf")
    for ep in range(1, args.episodes + 1):
        train_m = train_bc_episode(
            train_env, model, optimizer, device, args.side, max_steps=max_steps
        )
        val_m = evaluate_platform_bc(
            val_env, model, device, args.side, max_steps=max_steps
        )
        logger.log_episode(
            EpisodeMetrics(
                episode=ep,
                split="train",
                reward=0.0,
                hit_rate=train_m["bc_hit_rate"],
                steps=int(train_m["steps"]),
                epsilon=0.0,
                avg_loss=train_m["avg_loss"],
                buffer_size=0,
                global_step=ep,
            )
        )
        logger.log_episode(
            EpisodeMetrics(
                episode=ep,
                split="val",
                reward=0.0,
                hit_rate=val_m["bc_hit_rate"],
                steps=int(val_m.get("steps", 0)),
                epsilon=0.0,
                avg_loss=None,
                buffer_size=0,
                global_step=ep,
            )
        )
        print(
            f"ep {ep} train_bc_hit={train_m['bc_hit_rate']:.4f} "
            f"val_bc_hit={val_m['bc_hit_rate']:.4f} "
            f"val_worker_recall={val_m.get('worker_recall_at_k', 0):.4f}",
            flush=True,
        )
        if val_m["bc_hit_rate"] > best_val:
            best_val = val_m["bc_hit_rate"]
            ckpt = logger.checkpoint_path(f"{args.side}_best")
            torch.save(
                {
                    "policy": model.state_dict(),
                    "config": vars(
                        DQNConfig(
                            model_type=args.model,
                            device=args.device,
                            hidden_dim=args.hidden_dim,
                            extra_hidden_layers=args.extra_hidden_layers,
                            anchor_dim=anchor_dim,
                            candidate_dim=candidate_dim,
                        )
                    ),
                    "episode": ep,
                    "val_bc_hit_rate": best_val,
                },
                ckpt,
            )
            print(f"  -> 保存 {ckpt}", flush=True)

    final = logger.checkpoint_path(f"{args.side}_final")
    torch.save(
        {
            "policy": model.state_dict(),
            "config": vars(
                DQNConfig(
                    model_type=args.model,
                    device=args.device,
                    hidden_dim=args.hidden_dim,
                    extra_hidden_layers=args.extra_hidden_layers,
                    anchor_dim=anchor_dim,
                    candidate_dim=candidate_dim,
                )
            ),
        },
        final,
    )
    logger.save_summary()
    print(f"BC 完成: {final}", flush=True)


if __name__ == "__main__":
    main()
