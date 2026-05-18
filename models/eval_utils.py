"""评估循环（支持 DQN Agent 与基线 Policy）。"""

from __future__ import annotations

from typing import Callable, Protocol

from env.worker_env import Observation
from models.dqn import DQNAgent


class EnvProtocol(Protocol):
    config: object

    def reset(self) -> tuple[Observation, dict]: ...
    def step(self, action: int) -> tuple: ...


ActionFn = Callable[[Observation], int]


def run_eval_episode(
    env: EnvProtocol,
    select_action: ActionFn,
    *,
    max_steps: int | None = None,
) -> dict:
    obs, _ = env.reset()
    total_reward = 0.0
    hits = 0
    steps = 0
    limit = max_steps or getattr(env.config, "max_steps_per_episode", None)

    while True:
        action = select_action(obs)
        obs, reward, terminated, _, info = env.step(action)
        total_reward += reward
        hits += int(info.get("hit", False))
        steps += 1
        if terminated or (limit is not None and steps >= limit):
            break

    return {
        "reward": total_reward,
        "hit_rate": hits / max(steps, 1),
        "steps": steps,
        "hits": hits,
    }


def run_eval_full(
    env: EnvProtocol,
    select_action: ActionFn,
    *,
    max_steps: int | None = None,
) -> dict:
    """跑完整个 split 的所有事件（临时取消 episode 步数上限）。"""
    old_limit = getattr(env.config, "max_steps_per_episode", None)
    env.config.max_steps_per_episode = None  # type: ignore[attr-defined]
    try:
        return run_eval_episode(env, select_action, max_steps=max_steps)
    finally:
        env.config.max_steps_per_episode = old_limit  # type: ignore[attr-defined]


def dqn_action_fn(agent: DQNAgent) -> ActionFn:
    def _select(obs: Observation) -> int:
        return agent.select_action(obs, explore=False)

    return _select
