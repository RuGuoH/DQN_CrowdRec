"""Train dual DQN agents in the dynamic two-sided platform environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.platform_env import (
    PlatformSimulationEnv,
    add_platform_env_cli_args,
    platform_env_config_from_args,
)
from models.platform_training import run_platform_episode
from models.training_log import TrainingLogger
from src.config import Config, load_config
from src.dataset import build_dataset
from src.features import (
    PLATFORM_PROJECT_FEAT_DIM,
    REQUESTER_CONTEXT_FEAT_DIM,
    WORKER_FEAT_DIM,
)
from src.platform_dataset import PlatformDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dynamic platform DQN")
    parser.add_argument(
        "--max-projects",
        type=int,
        default=0,
        help="0=全量；调试可设 50/100",
    )
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--num-project-candidates", type=int, default=32)
    parser.add_argument("--num-worker-candidates", type=int, default=32)
    parser.add_argument("--include-truth-in-candidates", action="store_true")
    parser.add_argument("--project-wait-penalty", type=float, default=0.05)
    add_platform_env_cli_args(parser)
    parser.add_argument("--worker-pretrained", type=str, default=None)
    parser.add_argument("--requester-pretrained", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--requester-lr", type=float, default=None)
    parser.add_argument("--worker-replay-batch", type=int, default=64)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="若指定则同时覆盖 worker/requester replay batch（兼容旧 CLI）",
    )
    parser.add_argument("--requester-replay-batch", type=int, default=32)
    parser.add_argument("--requester-min-batch", type=int, default=8)
    parser.add_argument("--worker-replay-buffer", type=int, default=100_000)
    parser.add_argument("--requester-replay-buffer", type=int, default=50_000)
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="若指定则同时覆盖 worker/requester replay buffer（兼容旧 CLI）",
    )
    parser.add_argument("--target-update-freq", type=int, default=200)
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=120_000,
        help="worker ε 衰减步数（按梯度更新计）；全量 episode 下应跨多轮缓慢衰减",
    )
    parser.add_argument("--requester-epsilon-decay-steps", type=int, default=40_000)
    parser.add_argument("--epsilon-end", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="0=完整 episode；调试可设 100/800",
    )
    parser.add_argument("--update-every", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-dir", default="runs/platform")
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="默认 hidden size；可用 worker/requester 专用参数覆盖",
    )
    parser.add_argument("--worker-hidden-dim", type=int, default=None)
    parser.add_argument("--requester-hidden-dim", type=int, default=None)
    parser.add_argument(
        "--extra-hidden-layers",
        type=int,
        default=1,
        help="在旧网络结构基础上额外增加的隐藏层数",
    )
    parser.add_argument("--worker-extra-hidden-layers", type=int, default=None)
    parser.add_argument("--requester-extra-hidden-layers", type=int, default=None)
    parser.add_argument("--worker-model", choices=["dqn", "dueling"], default="dueling")
    parser.add_argument("--requester-model", choices=["dqn", "dueling"], default="dueling")
    parser.add_argument(
        "--worker-double-dqn",
        dest="worker_double_dqn",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-worker-double-dqn",
        dest="worker_double_dqn",
        action="store_false",
    )
    parser.add_argument(
        "--requester-double-dqn",
        dest="requester_double_dqn",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-requester-double-dqn",
        dest="requester_double_dqn",
        action="store_false",
    )
    args = parser.parse_args()

    worker_batch = (
        args.batch_size if args.batch_size is not None else args.worker_replay_batch
    )
    requester_batch = (
        args.batch_size
        if args.batch_size is not None
        else args.requester_replay_batch
    )
    worker_buffer = (
        args.buffer_size if args.buffer_size is not None else args.worker_replay_buffer
    )
    requester_buffer = (
        args.buffer_size
        if args.buffer_size is not None
        else args.requester_replay_buffer
    )
    requester_lr = args.lr if args.requester_lr is None else args.requester_lr
    worker_hidden_dim = (
        args.hidden_dim if args.worker_hidden_dim is None else args.worker_hidden_dim
    )
    requester_hidden_dim = (
        args.hidden_dim if args.requester_hidden_dim is None else args.requester_hidden_dim
    )
    worker_extra_hidden_layers = (
        args.extra_hidden_layers
        if args.worker_extra_hidden_layers is None
        else args.worker_extra_hidden_layers
    )
    requester_extra_hidden_layers = (
        args.extra_hidden_layers
        if args.requester_extra_hidden_layers is None
        else args.requester_extra_hidden_layers
    )

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
    env_cfg = platform_env_config_from_args(
        args,
        max_steps_per_episode=None if args.max_steps == 0 else args.max_steps,
    )

    train_env = PlatformSimulationEnv(train_platform, env_cfg, seed=42)
    val_env = PlatformSimulationEnv(val_platform, env_cfg, seed=2026)

    worker_cfg = DQNConfig(
        model_type=args.worker_model,
        double_dqn=args.worker_double_dqn,
        device=args.device,
        lr=args.lr,
        batch_size=worker_batch,
        min_batch_size=16,
        buffer_size=worker_buffer,
        target_update_freq=args.target_update_freq,
        epsilon_decay_steps=args.epsilon_decay_steps,
        epsilon_end=args.epsilon_end,
        hidden_dim=worker_hidden_dim,
        extra_hidden_layers=worker_extra_hidden_layers,
        anchor_dim=WORKER_FEAT_DIM,
        candidate_dim=PLATFORM_PROJECT_FEAT_DIM,
    )
    requester_cfg = DQNConfig(
        model_type=args.requester_model,
        double_dqn=args.requester_double_dqn,
        device=args.device,
        lr=requester_lr,
        batch_size=requester_batch,
        min_batch_size=args.requester_min_batch,
        buffer_size=requester_buffer,
        target_update_freq=args.target_update_freq,
        epsilon_decay_steps=args.requester_epsilon_decay_steps,
        epsilon_end=args.epsilon_end,
        hidden_dim=requester_hidden_dim,
        extra_hidden_layers=requester_extra_hidden_layers,
        anchor_dim=REQUESTER_CONTEXT_FEAT_DIM,
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

    if args.worker_pretrained:
        worker_agent.load(args.worker_pretrained, load_optimizer=False)
        worker_agent.sync_target()
        print(f"已加载 worker 预训练: {args.worker_pretrained}", flush=True)
    if args.requester_pretrained:
        requester_agent.load(args.requester_pretrained, load_optimizer=False)
        requester_agent.sync_target()
        print(f"已加载 requester 预训练: {args.requester_pretrained}", flush=True)

    truth_tag = "with_truth" if args.include_truth_in_candidates else "no_truth"
    recall_tag = "mixed" if not args.no_mixed_recall else "legacy"
    logger = TrainingLogger(
        Path(args.log_dir),
        run_name=f"platform_dqn_{args.reward_mode}_{truth_tag}_{recall_tag}",
    )
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

        val_score = validation_score(val_m)
        if val_score > best_val:
            best_val = val_score
            worker_agent.save_checkpoint(logger, "worker_best", extra={"episode": ep})
            requester_agent.save_checkpoint(
                logger,
                "requester_best",
                extra={"episode": ep},
            )
            print(
                f"  -> 新最佳 val_score={val_score:.4f} "
                f"(worker_U={val_m.get('avg_worker_utility', 0):.4f}, "
                f"requester_U={val_m.get('avg_requester_utility', 0):.4f}, "
                f"worker_hit={val_m['worker_hit_rate']:.4f}, "
                f"requester_hit={val_m['requester_hit_rate']:.4f})",
                flush=True,
            )

        if ep % args.save_every == 0:
            worker_agent.save_checkpoint(logger, f"worker_ep{ep:04d}")
            requester_agent.save_checkpoint(logger, f"requester_ep{ep:04d}")

    worker_agent.save_checkpoint(logger, "worker_final")
    requester_agent.save_checkpoint(logger, "requester_final")
    logger.save_summary()
    print(f"训练完成。指标: {logger.metrics_csv}", flush=True)


def validation_score(metrics: dict) -> float:
    """utility 模式按效用、hit 和 WAIT 风险选 checkpoint；legacy 仍参考 hit。"""
    if "avg_worker_utility" in metrics:
        return (
            metrics.get("avg_worker_utility", 0.0)
            + 5.0 * metrics.get("avg_requester_utility", 0.0)
            + 0.50 * metrics.get("worker_hit_rate", 0.0)
            + 2.00 * metrics.get("requester_hit_rate", 0.0)
            - 0.02 * metrics.get("avg_requester_pool_size", 0.0)
            - 0.001 * metrics.get("requester_decisions", 0.0)
            - 0.05 * metrics.get("avg_project_wait_days", 0.0)
        )
    return (
        metrics.get("worker_hit_rate", 0.0)
        + 5.0 * metrics.get("requester_hit_rate", 0.0)
        + 0.01 * metrics.get("worker_recall_at_k", 0.0)
    )


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
