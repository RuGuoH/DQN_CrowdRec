"""快速验证请求者侧环境 + DQN。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.requester_env import RequesterEnvConfig, RequesterRecommendationEnv
from models.dqn import DQNAgent, DQNConfig
from src.config import Config, load_config
from src.dataset import build_dataset
from src.features import PROJECT_FEAT_DIM, WORKER_FEAT_DIM


def main() -> None:
    cfg = load_config()
    cfg = Config(
        cfg.data_dir, cfg.min_start_date, cfg.page_limit,
        cfg.train_ratio, cfg.val_ratio, 50, cfg.cache_dir,
    )
    ds = build_dataset(cfg)
    env = RequesterRecommendationEnv(
        ds, "train", RequesterEnvConfig(num_candidates=8)
    )
    agent = DQNAgent(
        8,
        DQNConfig(
            batch_size=8,
            buffer_size=64,
            anchor_dim=PROJECT_FEAT_DIM,
            candidate_dim=WORKER_FEAT_DIM,
        ),
    )
    obs, info = env.reset()
    print("events:", len(env.events), "project:", info.get("project_id"))

    t0 = time.perf_counter()
    for _ in range(50):
        a = agent.select_action(obs, explore=False)
        obs, r, done, _, step_info = env.step(a)
        if done:
            obs, _ = env.reset()
    print(f"50 steps in {time.perf_counter()-t0:.2f}s, hit={step_info.get('hit')}")


if __name__ == "__main__":
    main()
