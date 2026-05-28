"""Training and evaluation loops for the dynamic platform environment."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from env.platform_env import ActorName, PlatformDecision, PlatformSimulationEnv
from env.worker_env import Observation

if TYPE_CHECKING:
    from models.dqn import DQNAgent

ActionSelector = Callable[[PlatformSimulationEnv, PlatformDecision], int]


def run_platform_episode(
    env: PlatformSimulationEnv,
    worker_agent: "DQNAgent",
    requester_agent: "DQNAgent",
    *,
    train: bool = True,
    update_every: int = 4,
    max_steps: int | None = None,
) -> dict:
    decision = env.reset()
    pending: dict[ActorName, tuple[Observation, int, float]] = {}
    losses: dict[ActorName, list[float]] = {"worker": [], "requester": []}
    observe_counts: dict[ActorName, int] = {"worker": 0, "requester": 0}
    limit = max_steps or env.config.max_steps_per_episode
    steps = 0

    while decision is not None:
        actor = decision.actor
        obs = decision.observation
        agent = worker_agent if actor == "worker" else requester_agent

        if train and actor in pending:
            _observe_pending(
                agent,
                actor,
                pending.pop(actor),
                obs,
                False,
                losses,
                observe_counts,
                update_every,
            )

        action = agent.select_action(obs, explore=train)
        step = env.step(action)
        if train:
            pending[actor] = (obs, action, step.reward)

        steps += 1
        if step.terminated or (limit is not None and steps >= limit):
            if train:
                for pending_actor, payload in list(pending.items()):
                    pending_agent = (
                        worker_agent
                        if pending_actor == "worker"
                        else requester_agent
                    )
                    _observe_pending(
                        pending_agent,
                        pending_actor,
                        payload,
                        env.empty_observation(pending_actor),
                        True,
                        losses,
                        observe_counts,
                        update_every,
                    )
                pending.clear()
            break
        decision = step.decision

    metrics = env.final_metrics()
    metrics["worker_avg_loss"] = _avg(losses["worker"])
    metrics["requester_avg_loss"] = _avg(losses["requester"])
    return metrics


def run_platform_eval(
    env: PlatformSimulationEnv,
    worker_select: ActionSelector,
    requester_select: ActionSelector,
    *,
    max_steps: int | None = None,
) -> dict:
    decision = env.reset()
    limit = max_steps or env.config.max_steps_per_episode
    steps = 0

    while decision is not None:
        if decision.actor == "worker":
            action = worker_select(env, decision)
        else:
            action = requester_select(env, decision)
        step = env.step(action)
        steps += 1
        if step.terminated or (limit is not None and steps >= limit):
            break
        decision = step.decision
    return env.final_metrics()


def _observe_pending(
    agent: "DQNAgent",
    actor: ActorName,
    pending: tuple[Observation, int, float],
    next_obs: Observation,
    done: bool,
    losses: dict[ActorName, list[float]],
    observe_counts: dict[ActorName, int],
    update_every: int,
) -> None:
    from models.dqn import Transition

    obs, action, reward = pending
    agent.observe(
        Transition(
            obs=obs.to_dict(),
            action=action,
            reward=reward,
            next_obs=next_obs.to_dict(),
            done=done,
        )
    )
    observe_counts[actor] += 1
    if observe_counts[actor] % update_every == 0:
        loss = agent.update()
        if loss is not None:
            losses[actor].append(loss)


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
