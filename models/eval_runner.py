"""评估入口逻辑（供 evaluate.py / run_baselines.py 共用）。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch

from env.requester_env import RequesterEnvConfig, RequesterRecommendationEnv
from env.worker_env import EnvConfig, WorkerRecommendationEnv
from models.baselines import make_baseline
from models.dqn import DQNAgent, DQNConfig
from models.eval_utils import dqn_action_fn, run_eval_full
from src.dataset import CrowdsourcingDataset, SplitName
from src.features import PROJECT_FEAT_DIM, WORKER_FEAT_DIM


def build_env(
    side: str,
    ds: CrowdsourcingDataset,
    split: SplitName,
    num_candidates: int,
    seed: int,
):
    if side == "worker":
        cfg = EnvConfig(num_candidates=num_candidates)
        return WorkerRecommendationEnv(ds, split=split, config=cfg, seed=seed)
    cfg = RequesterEnvConfig(num_candidates=num_candidates)
    return RequesterRecommendationEnv(ds, split=split, config=cfg, seed=seed)


def load_dqn_agent(checkpoint: Path, num_candidates: int, side: str) -> DQNAgent:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    fields = DQNConfig.__dataclass_fields__
    dqn_cfg = DQNConfig(**{k: v for k, v in cfg_dict.items() if k in fields})
    if side == "worker":
        dqn_cfg.anchor_dim = WORKER_FEAT_DIM
        dqn_cfg.candidate_dim = PROJECT_FEAT_DIM
    else:
        dqn_cfg.anchor_dim = PROJECT_FEAT_DIM
        dqn_cfg.candidate_dim = WORKER_FEAT_DIM
    agent = DQNAgent(num_actions=num_candidates, config=dqn_cfg)
    agent.load(checkpoint, load_optimizer=False)
    return agent


def evaluate_one(
    side: str,
    ds: CrowdsourcingDataset,
    split: SplitName,
    *,
    policy: str,
    checkpoint: Path | None,
    num_candidates: int,
    max_steps: int | None,
    seed: int,
) -> dict:
    env = build_env(side, ds, split, num_candidates, seed)

    if policy == "dqn":
        if checkpoint is None:
            raise ValueError("评估 DQN 需指定 checkpoint")
        agent = load_dqn_agent(checkpoint, num_candidates, side)
        select = dqn_action_fn(agent)
        policy_name = f"dqn:{checkpoint.parent.parent.name}"
    else:
        baseline = make_baseline(policy, side, seed=seed)

        def select(obs):
            return baseline.select_action(env, obs)

        policy_name = policy

    metrics = run_eval_full(env, select, max_steps=max_steps)
    return {
        "side": side,
        "split": split,
        "policy": policy_name,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "num_candidates": num_candidates,
        "num_events": len(env.events),
        **metrics,
        "evaluated_at": datetime.now().isoformat(),
    }
