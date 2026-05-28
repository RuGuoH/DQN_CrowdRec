"""Baselines for the dynamic two-sided platform simulation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from env.platform_env import PlatformDecision, PlatformSimulationEnv

PlatformSelector = Callable[[PlatformSimulationEnv, PlatformDecision], int]

WORKER_PLATFORM_BASELINES = [
    "random_project",
    "popularity",
    "category_match",
    "industry_match",
    "award",
    "low_wait_project",
    "joint_heuristic",
]

REQUESTER_PLATFORM_BASELINES = [
    "wait_until_deadline",
    "select_first",
    "worker_quality",
    "worker_activity",
    "worker_category_match",
    "worker_industry_match",
]


def select_worker_project(
    name: str,
    env: PlatformSimulationEnv,
    decision: PlatformDecision,
) -> int:
    valid = np.flatnonzero(decision.observation.action_mask)
    if len(valid) == 0:
        return 0
    project_ids = decision.info.get("candidate_project_ids", [])
    worker_id = int(decision.info["worker_id"])
    timestamp = env.current_time_or_project_time(project_ids[0]) if project_ids else None

    if name == "random_project":
        return int(env.rng.choice(valid))
    if name == "popularity":
        return _best(valid, lambda a: env.dataset.projects[project_ids[a]].entry_count)
    if name == "award":
        return _best(valid, lambda a: env.dataset.projects[project_ids[a]].total_awards)
    if name == "low_wait_project":
        return _best(valid, lambda a: env._project_wait_days(project_ids[a], timestamp))
    if name in {"category_match", "joint_heuristic"}:
        profile = env.encoder.worker_history_profile(worker_id, timestamp)
        return _best(
            valid,
            lambda a: (
                1.0
                if profile.dominant_category
                == env.dataset.projects[project_ids[a]].category
                else 0.0
            )
            + (0.01 * env._project_wait_days(project_ids[a], timestamp))
            + (0.0001 * env.dataset.projects[project_ids[a]].total_awards),
        )
    if name == "industry_match":
        profile = env.encoder.worker_history_profile(worker_id, timestamp)
        return _best(
            valid,
            lambda a: (
                1.0
                if profile.dominant_industry_id
                == env.dataset.projects[project_ids[a]].industry_id
                else 0.0
            )
            + (0.01 * env._project_wait_days(project_ids[a], timestamp)),
        )
    raise KeyError(f"unknown worker platform baseline: {name}")


def select_requester_worker(
    name: str,
    env: PlatformSimulationEnv,
    decision: PlatformDecision,
) -> int:
    valid = np.flatnonzero(decision.observation.action_mask)
    if len(valid) == 0:
        return 0
    worker_ids = decision.info.get("candidate_worker_ids", [])
    project_id = int(decision.info["project_id"])
    project = env.dataset.projects[project_id]
    timestamp = env.current_time_or_project_time(project_id)

    if name == "wait_until_deadline" and 0 in valid:
        return 0
    non_wait = [a for a in valid if a != 0 and a < len(worker_ids)]
    if not non_wait:
        return int(valid[0])
    if name in {"wait_until_deadline", "select_first"}:
        return int(non_wait[0])
    if name == "worker_quality":
        return _best(non_wait, lambda a: env.dataset.get_worker_quality(worker_ids[a]))
    if name == "worker_activity":
        return _best(
            non_wait,
            lambda a: env.encoder.worker_history_profile(
                worker_ids[a],
                timestamp,
            ).past_count,
        )
    if name == "worker_category_match":
        return _best(
            non_wait,
            lambda a: _match_score(
                env.encoder.worker_history_profile(worker_ids[a], timestamp).dominant_category,
                project.category,
                env.dataset.get_worker_quality(worker_ids[a]),
            ),
        )
    if name == "worker_industry_match":
        return _best(
            non_wait,
            lambda a: _match_score(
                env.encoder.worker_history_profile(
                    worker_ids[a],
                    timestamp,
                ).dominant_industry_id,
                project.industry_id,
                env.dataset.get_worker_quality(worker_ids[a]),
            ),
        )
    raise KeyError(f"unknown requester platform baseline: {name}")


def make_platform_selectors(
    worker_policy: str,
    requester_policy: str,
) -> tuple[PlatformSelector, PlatformSelector]:
    def _worker(env: PlatformSimulationEnv, decision: PlatformDecision) -> int:
        return select_worker_project(worker_policy, env, decision)

    def _requester(env: PlatformSimulationEnv, decision: PlatformDecision) -> int:
        return select_requester_worker(requester_policy, env, decision)

    return _worker, _requester


def _best(actions, score_fn) -> int:
    return int(max(actions, key=lambda a: (score_fn(a), -int(a))))


def _match_score(left, right, quality: float) -> float:
    return (1.0 if left == right else 0.0) + quality * 0.01
