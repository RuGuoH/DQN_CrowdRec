"""Train dual DQN agents in the dynamic two-sided platform environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.platform_env import PlatformEnvConfig, PlatformSimulationEnv
from models.platform_training import run_platform_episode
from models.training_log import TrainingLogger
from src.config import Config, load_config
from src.dataset import build_dataset
from src.features import PROJECT_FEAT_DIM, WORKER_FEAT_DIM
from src.platform_dataset import PlatformDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dynamic platform DQN")
    parser.add_argument("--max-projects", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--num-project-candidates", type=int, default=32)
    parser.add_argument("--num-worker-candidates", type=int, default=32)
    parser.add_argument("--worker-model", choices=["dqn", "dueling"], default="dqn")
    parser.add_argument("--requester-model", choices=["dqn", "dueling"], default="dqn")
    parser.add_argument("--worker-double-dqn", action="store_true")
    parser.add_argument("--requester-double-dqn", action="store_true")
    parser.add_argument("--include-truth-in-candidates", action="store_true")
    parser.add_argument("--project-wait-penalty", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--target-update-freq", type=int, default=200)
    parser.add_argument("--epsilon-decay-steps", type=int, default=1_000)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--update-every", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-dir", default="runs/platform")
    args = parser.parse_args()

    try:
        from models.dqn import DQNAgent, DQNConfig
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "PyTorch is required for DQN training. Install requirements or run "
                "this script in the project training environment."
            ) from exc
        raise

    ds = build_with_limit(args.max_projects)
    train_platform = PlatformDataset(ds, "train")
    val_platform = PlatformDataset(ds, "val")
    env_cfg = PlatformEnvConfig(
        num_project_candidates=args.num_project_candidates,
        num_worker_candidates=args.num_worker_candidates,
        project_wait_penalty=args.project_wait_penalty,
        include_truth_in_candidates=args.include_truth_in_candidates,
        max_steps_per_episode=None if args.max_steps == 0 else args.max_steps,
    )

    train_env = PlatformSimulationEnv(train_platform, env_cfg, seed=42)
    val_env = PlatformSimulationEnv(val_platform, env_cfg, seed=2026)

    worker_cfg = DQNConfig(
        model_type=args.worker_model,
        double_dqn=args.worker_double_dqn,
        device=args.device,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        target_update_freq=args.target_update_freq,
        epsilon_decay_steps=args.epsilon_decay_steps,
        epsilon_end=args.epsilon_end,
        anchor_dim=WORKER_FEAT_DIM,
        candidate_dim=PROJECT_FEAT_DIM,
    )
    requester_cfg = DQNConfig(
        model_type=args.requester_model,
        double_dqn=args.requester_double_dqn,
        device=args.device,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        target_update_freq=args.target_update_freq,
        epsilon_decay_steps=args.epsilon_decay_steps,
        epsilon_end=args.epsilon_end,
        anchor_dim=PROJECT_FEAT_DIM,
        candidate_dim=WORKER_FEAT_DIM,
    )
    worker_agent = DQNAgent(
        num_actions=args.num_project_candidates,
        config=worker_cfg,
    )
    requester_agent = DQNAgent(
        num_actions=args.num_worker_candidates + 1,
        config=requester_cfg,
    )

    truth_tag = "with_truth" if args.include_truth_in_candidates else "no_truth"
    logger = TrainingLogger(Path(args.log_dir), run_name=f"platform_dqn_{truth_tag}")
    logger.save_config(
        {
            "dataset": ds.summary(),
            "train_platform": train_platform.summary(),
            "val_platform": val_platform.summary(),
            "env": vars(env_cfg),
            "worker_dqn": vars(worker_cfg),
            "requester_dqn": vars(requester_cfg),
            "episodes": args.episodes,
        }
    )
    print(f"日志目录: {logger.run_dir}", flush=True)

    best_val = float("-inf")
    for ep in range(1, args.episodes + 1):
        train_m = run_platform_episode(
            train_env,
            worker_agent,
            requester_agent,
            train=True,
            update_every=args.update_every,
        )
        log_metrics(logger, ep, "train", train_m, worker_agent, requester_agent)

        val_m = run_platform_episode(
            val_env,
            worker_agent,
            requester_agent,
            train=False,
        )
        log_metrics(logger, ep, "val", val_m, worker_agent, requester_agent)

        if val_m["platform_reward"] > best_val:
            best_val = val_m["platform_reward"]
            worker_agent.save_checkpoint(logger, "worker_best", extra={"episode": ep})
            requester_agent.save_checkpoint(
                logger,
                "requester_best",
                extra={"episode": ep},
            )
            print(f"  -> 新最佳 val platform_reward={best_val:.3f}", flush=True)

        if ep % args.save_every == 0:
            worker_agent.save_checkpoint(logger, f"worker_ep{ep:04d}")
            requester_agent.save_checkpoint(logger, f"requester_ep{ep:04d}")

    worker_agent.save_checkpoint(logger, "worker_final")
    requester_agent.save_checkpoint(logger, "requester_final")
    logger.save_summary()
    print(f"训练完成。指标: {logger.metrics_csv}", flush=True)


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


def log_metrics(
    logger: TrainingLogger,
    episode: int,
    split: str,
    metrics: dict,
    worker_agent: DQNAgent,
    requester_agent: DQNAgent,
) -> None:
    logger.log_dict(
        {
            "episode": episode,
            "split": split,
            **metrics,
            "worker_epsilon": worker_agent.epsilon,
            "requester_epsilon": requester_agent.epsilon,
            "worker_buffer_size": len(worker_agent.replay),
            "requester_buffer_size": len(requester_agent.replay),
            "worker_global_step": worker_agent.global_step,
            "requester_global_step": requester_agent.global_step,
        }
    )


if __name__ == "__main__":
    main()
