"""训练参与者侧 DQN。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.worker_env import EnvConfig, WorkerRecommendationEnv
from models.dqn import DQNAgent, DQNConfig, save_best_checkpoint
from models.train_utils import run_episode
from models.training_log import EpisodeMetrics, TrainingLogger
from src.config import Config, load_config
from src.dataset import build_dataset
from src.features import PROJECT_FEAT_DIM, WORKER_FEAT_DIM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-projects", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--num-candidates", type=int, default=32)
    parser.add_argument("--model", choices=["dqn", "dueling"], default="dqn")
    parser.add_argument("--double-dqn", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--log-dir", default="runs/worker")
    parser.add_argument("--save-every", type=int, default=5, help="每 N 个 episode 存一次")
    parser.add_argument("--update-every", type=int, default=4)
    parser.add_argument(
    "--no-truth-in-candidates",
    action="store_true",
    help="Do not force ground-truth project into candidate set.",
)
    args = parser.parse_args()

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
    train_env = WorkerRecommendationEnv(ds, split="train", config=env_cfg)
    val_env = WorkerRecommendationEnv(ds, split="val", config=env_cfg)

    dqn_cfg = DQNConfig(
        model_type=args.model,
        double_dqn=args.double_dqn,
        device=args.device,
        batch_size=32,
        buffer_size=10_000,
        anchor_dim=WORKER_FEAT_DIM,
        candidate_dim=PROJECT_FEAT_DIM,
    )
    agent = DQNAgent(num_actions=env_cfg.num_candidates, config=dqn_cfg)
    run_suffix = "no_truth" if args.no_truth_in_candidates else "with_truth"
    logger = TrainingLogger(
    log_dir=Path(args.log_dir),
    run_name=f"worker_dqn_{run_suffix}",
)
    logger.save_config(
        {
            "side": "worker",
            "dataset": ds.summary(),
            "env": vars(env_cfg),
            "dqn": vars(dqn_cfg),
            "episodes": args.episodes,
        }
    )
    print(f"日志目录: {logger.run_dir}", flush=True)

    best_val_hit = float("-inf")
    for ep in range(1, args.episodes + 1):
        train_m = run_episode(
            train_env, agent, train=True, update_every=args.update_every
        )
        val_m = run_episode(val_env, agent, train=False)

        logger.log_episode(
            EpisodeMetrics(
                episode=ep,
                split="train",
                reward=train_m["reward"],
                hit_rate=train_m["hit_rate"],
                steps=train_m["steps"],
                epsilon=agent.epsilon,
                avg_loss=train_m["avg_loss"],
                buffer_size=len(agent.replay),
                global_step=agent.global_step,
            )
        )
        logger.log_episode(
            EpisodeMetrics(
                episode=ep,
                split="val",
                reward=val_m["reward"],
                hit_rate=val_m["hit_rate"],
                steps=val_m["steps"],
                epsilon=agent.epsilon,
                avg_loss=None,
                buffer_size=len(agent.replay),
                global_step=agent.global_step,
            )
        )

        best_val_hit, improved = save_best_checkpoint(
            agent,
            logger,
            val_m["hit_rate"],
            best_val_hit,
            extra={"episode": ep, "val_hit_rate": val_m["hit_rate"]},
        )
        if improved:
            print(f"  -> 新最佳 val hit={val_m['hit_rate']:.3f} 已保存 best.pt", flush=True)

        if ep % args.save_every == 0:
            agent.save_checkpoint(
                logger,
                f"ep{ep:04d}",
                extra={"episode": ep, "val_hit_rate": val_m["hit_rate"]},
            )

    agent.save_checkpoint(logger, "final")
    logger.save_summary()
    print(f"训练完成。指标: {logger.metrics_csv}", flush=True)


if __name__ == "__main__":
    main()
