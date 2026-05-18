"""训练循环工具函数。"""

from __future__ import annotations

from typing import Protocol

from models.dqn import DQNAgent, Transition


class EnvProtocol(Protocol):
    config: object

    def reset(self) -> tuple: ...
    def step(self, action: int) -> tuple: ...


def run_episode(
    env: EnvProtocol,
    agent: DQNAgent,
    *,
    train: bool = True,
    update_every: int = 4,
) -> dict:
    obs, _ = env.reset()
    total_reward = 0.0
    hits = 0
    steps = 0
    losses: list[float] = []
    max_steps = getattr(env.config, "max_steps_per_episode", None)

    while True:
        action = agent.select_action(obs, explore=train)
        next_obs, reward, terminated, _, info = env.step(action)
        total_reward += reward
        hits += int(info.get("hit", False))
        steps += 1

        if train:
            agent.observe(
                Transition(
                    obs=obs.to_dict(),
                    action=action,
                    reward=reward,
                    next_obs=next_obs.to_dict(),
                    done=terminated,
                )
            )
            if steps % update_every == 0:
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)

        if terminated or (max_steps is not None and steps >= max_steps):
            break
        obs = next_obs

    return {
        "reward": total_reward,
        "hit_rate": hits / max(steps, 1),
        "steps": steps,
        "avg_loss": sum(losses) / len(losses) if losses else None,
    }
