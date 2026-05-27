"""Behavior Cloning (BC) 预训练脚本。

目标:
- 使用历史行为监督学习初始化 Q 网络
- 再将预训练参数用于 DQN 微调

训练目标:
- 输入: observation
- 标签: ground-truth action
- loss: cross entropy

示例:

# worker side
python scripts/pretrain_bc.py \
    --side worker \
    --episodes 5

# requester side
python scripts/pretrain_bc.py \
    --side requester \
    --model dueling

# no_truth 消融
python scripts/pretrain_bc.py \
    --side worker \
    --no-truth-in-candidates
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

from env.requester_env import RequesterRecommendationEnv
from env.worker_env import EnvConfig, WorkerRecommendationEnv
from models.dqn import DQNConfig, build_q_network
from models.training_log import EpisodeMetrics, TrainingLogger
from src.config import Config, load_config
from src.dataset import build_dataset
from src.features import PROJECT_FEAT_DIM, WORKER_FEAT_DIM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate_bc(
    env,
    model,
    device: torch.device,
    max_steps: int | None = None,
) -> dict:
    """简单验证 accuracy / reward。"""

    obs,_ = env.reset()

    total_reward = 0.0
    hits = 0
    steps = 0

    while True:
        worker_feat = torch.as_tensor(
            obs.worker_feat,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        candidate_feat = torch.as_tensor(
            obs.candidate_feat,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        action_mask = torch.as_tensor(
            obs.action_mask,
            device=device,
        ).unsqueeze(0)

        q = model(worker_feat, candidate_feat, action_mask)
        action = int(q.argmax(dim=1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        total_reward += reward
        hits += int(info.get("hit", False))
        steps += 1

        obs = next_obs

        if done:
            break

        if max_steps is not None and steps >= max_steps:
            break

    return {
        "reward": total_reward,
        "hit_rate": hits / max(steps, 1),
        "steps": steps,
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="BC pretraining")

    parser.add_argument(
        "--side",
        choices=["worker", "requester"],
        required=True,
    )

    parser.add_argument("--max-projects", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--num-candidates", type=int, default=32)
    parser.add_argument("--model", choices=["dqn", "dueling"], default="dqn")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default="runs/bc")

    parser.add_argument(
        "--no-truth-in-candidates",
        action="store_true",
        help="Do not force ground-truth item into candidate set.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    cfg = load_config()

    if args.max_projects > 0:
        cfg = Config(
            data_dir=cfg.data_dir,
            min_start_date=cfg.min_start_date,
            page_limit=cfg.page_limit,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            max_projects=args.max_projects,
            cache_dir=cfg.cache_dir,
        )

    ds = build_dataset(cfg)
    print("数据:", ds.summary(), flush=True)

    max_steps = None if args.max_steps == 0 else args.max_steps

    env_cfg = EnvConfig(
        num_candidates=args.num_candidates,
        max_steps_per_episode=max_steps,
        include_truth_in_candidates=not args.no_truth_in_candidates,
    )

    if args.side == "worker":
        train_env = WorkerRecommendationEnv(ds, split="train", config=env_cfg)
        val_env = WorkerRecommendationEnv(ds, split="val", config=env_cfg)

        anchor_dim = WORKER_FEAT_DIM
        candidate_dim = PROJECT_FEAT_DIM

    else:
        train_env = RequesterRecommendationEnv(ds, split="train", config=env_cfg)
        val_env = RequesterRecommendationEnv(ds, split="val", config=env_cfg)

        anchor_dim = PROJECT_FEAT_DIM
        candidate_dim = WORKER_FEAT_DIM

    dqn_cfg = DQNConfig(
        model_type=args.model,
        device=args.device,
        hidden_dim=args.hidden_dim,
        anchor_dim=anchor_dim,
        candidate_dim=candidate_dim,
    )

    device = torch.device(args.device)

    model = build_q_network(
        model_type=args.model,
        num_actions=args.num_candidates,
        hidden_dim=args.hidden_dim,
        anchor_dim=anchor_dim,
        candidate_dim=candidate_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_suffix = "no_truth" if args.no_truth_in_candidates else "with_truth"

    logger = TrainingLogger(
        log_dir=Path(args.log_dir),
        run_name=f"bc_{args.side}_{args.model}_{run_suffix}",
    )

    logger.save_config(
        {
            "side": args.side,
            "dataset": ds.summary(),
            "env": vars(env_cfg),
            "bc": {
                "model": args.model,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "hidden_dim": args.hidden_dim,
            },
            "episodes": args.episodes,
        }
    )

    print(f"日志目录: {logger.run_dir}", flush=True)

    best_val_hit = float("-inf")

    for ep in range(1, args.episodes + 1):
        model.train()

        obs,_ = train_env.reset()

        total_loss = 0.0
        total_reward = 0.0
        hits = 0
        steps = 0

        while True:
            action = train_env.optimal_action(obs)

            if action is None:
                     next_obs, reward, terminated, truncated, info = train_env.step(action)

                     done = terminated or truncated
                     obs = next_obs

                     if done:
                        break

                     continue
            worker_feat = torch.as_tensor(
                obs.worker_feat,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            candidate_feat = torch.as_tensor(
                obs.candidate_feat,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            action_mask = torch.as_tensor(
                obs.action_mask,
                device=device,
            ).unsqueeze(0)

            logits = model(worker_feat, candidate_feat, action_mask)

            target = torch.tensor([action], dtype=torch.long, device=device)

            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            next_obs, reward, terminated, truncated, info = train_env.step(action)

            done = terminated or truncated

            total_loss += float(loss.item())
            total_reward += reward
            hits += int(info.get("hit", False))
            steps += 1

            obs = next_obs

            if done:
                break

            if max_steps is not None and steps >= max_steps:
                break

        avg_loss = total_loss / max(steps, 1)

        train_metrics = {
            "reward": total_reward,
            "hit_rate": hits / max(steps, 1),
            "steps": steps,
            "avg_loss": avg_loss,
        }

        val_metrics = evaluate_bc(
            env=val_env,
            model=model,
            device=device,
            max_steps=max_steps,
        )

        logger.log_episode(
            EpisodeMetrics(
                episode=ep,
                split="train",
                reward=train_metrics["reward"],
                hit_rate=train_metrics["hit_rate"],
                steps=train_metrics["steps"],
                epsilon=0.0,
                avg_loss=train_metrics["avg_loss"],
                buffer_size=0,
                global_step=ep,
            )
        )

        logger.log_episode(
            EpisodeMetrics(
                episode=ep,
                split="val",
                reward=val_metrics["reward"],
                hit_rate=val_metrics["hit_rate"],
                steps=val_metrics["steps"],
                epsilon=0.0,
                avg_loss=None,
                buffer_size=0,
                global_step=ep,
            )
        )

        if val_metrics["hit_rate"] > best_val_hit:
            best_val_hit = val_metrics["hit_rate"]

            ckpt_path = logger.checkpoint_path("best")

            torch.save(
                {
                    "policy": model.state_dict(),
                    "config": vars(dqn_cfg),
                    "episode": ep,
                    "val_hit_rate": best_val_hit,
                },
                ckpt_path,
            )

            print(
                f"  -> 新最佳 BC checkpoint 已保存: {ckpt_path}",
                flush=True,
            )

    final_path = logger.checkpoint_path("final")

    torch.save(
        {
            "policy": model.state_dict(),
            "config": vars(dqn_cfg),
        },
        final_path,
    )

    logger.save_summary()

    print(f"\nBC 预训练完成。")
    print(f"Final checkpoint: {final_path}")
    print(f"Metrics: {logger.metrics_csv}")


if __name__ == "__main__":
    main()
