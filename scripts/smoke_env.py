"""快速验证 MDP 环境与 DQN 前向（不跑完整训练）。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.worker_env import EnvConfig, WorkerRecommendationEnv
from models.dqn import DQNAgent, DQNConfig
from src.config import Config, load_config
from src.dataset import build_dataset


def main() -> None:
    cfg = load_config()
    cfg = Config(
        data_dir=cfg.data_dir,
        min_start_date=cfg.min_start_date,
        page_limit=cfg.page_limit,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        max_projects=50,
        cache_dir=cfg.cache_dir,
    )
    ds = build_dataset(cfg)
    env = WorkerRecommendationEnv(ds, "train", EnvConfig(num_candidates=8))
    agent = DQNAgent(8, DQNConfig(batch_size=8, buffer_size=64))

    obs, info = env.reset()
    print("events:", len(env.events), "first:", info.get("truth_project_id"))

    t0 = time.perf_counter()
    for i in range(50):
        action = agent.select_action(obs, explore=False)
        obs, reward, done, _, step_info = env.step(action)
        if done:
            obs, _ = env.reset()
    elapsed = time.perf_counter() - t0
    print(f"50 steps in {elapsed:.2f}s ({elapsed/50*1000:.1f} ms/step)")
    print("last hit:", step_info.get("hit"), "reward:", step_info.get("reward"))


if __name__ == "__main__":
    main()
