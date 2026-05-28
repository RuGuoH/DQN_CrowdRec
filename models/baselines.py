"""众包推荐基线策略（参与者侧 / 请求者侧）。"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from env.requester_env import RequesterRecommendationEnv
from env.worker_env import Observation, WorkerRecommendationEnv

EnvT = WorkerRecommendationEnv | RequesterRecommendationEnv


class Policy(Protocol):
    name: str

    def select_action(self, env: EnvT, obs: Observation) -> int: ...


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, env: EnvT, obs: Observation) -> int:
        valid = np.flatnonzero(obs.action_mask)
        if len(valid) == 0:
            return 0
        return int(self.rng.choice(valid))


class PopularityPolicy:
    """参与者侧：选当前投稿数最多的项目。"""

    name = "popularity"

    def select_action(self, env: EnvT, obs: Observation) -> int:
        if not isinstance(env, WorkerRecommendationEnv):
            raise TypeError("PopularityPolicy 仅用于 worker 环境")
        valid = np.flatnonzero(obs.action_mask)
        if len(valid) == 0:
            return 0
        best_a, best_cnt = valid[0], -1
        for a in valid:
            pid = env._candidate_ids[a]
            cnt = env.dataset.projects[pid].entry_count
            if cnt > best_cnt:
                best_cnt, best_a = cnt, a
        return int(best_a)


class CategoryMatchPolicy:
    """参与者侧：选类目与 worker 历史主类目一致的项目。"""

    name = "category_match"

    def select_action(self, env: EnvT, obs: Observation) -> int:
        if not isinstance(env, WorkerRecommendationEnv):
            raise TypeError("CategoryMatchPolicy 仅用于 worker 环境")
        valid = np.flatnonzero(obs.action_mask)
        if len(valid) == 0:
            return 0
        assert env._current is not None
        profile = env.encoder.worker_history_profile(
            env._current.worker_id,
            env._current.timestamp,
        )
        dom = profile.dominant_category
        if dom is None:
            return int(valid[0])
        for a in valid:
            pid = env._candidate_ids[a]
            if env.dataset.projects[pid].category == dom:
                return int(a)
        return int(valid[0])


class IndustryMatchPolicy:
    """参与者侧：选行业与 worker 历史主行业一致的项目。"""

    name = "industry_match"

    def select_action(self, env: EnvT, obs: Observation) -> int:
        if not isinstance(env, WorkerRecommendationEnv):
            raise TypeError("IndustryMatchPolicy 仅用于 worker 环境")
        valid = np.flatnonzero(obs.action_mask)
        if len(valid) == 0:
            return 0
        assert env._current is not None
        profile = env.encoder.worker_history_profile(
            env._current.worker_id,
            env._current.timestamp,
        )
        dom = profile.dominant_industry_id
        if dom is None:
            return int(valid[0])
        for a in valid:
            pid = env._candidate_ids[a]
            if env.dataset.projects[pid].industry_id == dom:
                return int(a)
        return int(valid[0])


class AwardPolicy:
    """参与者侧：选奖金最高的开放项目。"""

    name = "award"

    def select_action(self, env: EnvT, obs: Observation) -> int:
        if not isinstance(env, WorkerRecommendationEnv):
            raise TypeError("AwardPolicy 仅用于 worker 环境")
        valid = np.flatnonzero(obs.action_mask)
        if len(valid) == 0:
            return 0
        best_a, best_award = valid[0], -1.0
        for a in valid:
            pid = env._candidate_ids[a]
            award = env.dataset.projects[pid].total_awards
            if award > best_award:
                best_award, best_a = award, a
        return int(best_a)


class WorkerQualityPolicy:
    """请求者侧：选质量分最高的 worker。"""

    name = "worker_quality"

    def select_action(self, env: EnvT, obs: Observation) -> int:
        if not isinstance(env, RequesterRecommendationEnv):
            raise TypeError("WorkerQualityPolicy 仅用于 requester 环境")
        valid = np.flatnonzero(obs.action_mask)
        if len(valid) == 0:
            return 0
        best_a, best_q = valid[0], -1.0
        for a in valid:
            wid = env._candidates[a]
            q = env.dataset.get_worker_quality(wid)
            if q > best_q:
                best_q, best_a = q, a
        return int(best_a)


class WorkerActivityPolicy:
    """请求者侧：选历史投稿次数最多的 worker。"""

    name = "worker_activity"

    def select_action(self, env: EnvT, obs: Observation) -> int:
        if not isinstance(env, RequesterRecommendationEnv):
            raise TypeError("WorkerActivityPolicy 仅用于 requester 环境")
        valid = np.flatnonzero(obs.action_mask)
        if len(valid) == 0:
            return 0
        assert env._current is not None
        t = env._current.timestamp
        best_a, best_n = valid[0], -1
        for a in valid:
            wid = env._candidates[a]
            n = len(env.encoder._past_entries(wid, t))
            if n > best_n:
                best_n, best_a = n, a
        return int(best_a)


class RequesterCategoryMatchPolicy:
    """请求者侧：选历史主类目与当前项目类目一致的 worker。"""

    name = "worker_category_match"

    def select_action(self, env: EnvT, obs: Observation) -> int:
        if not isinstance(env, RequesterRecommendationEnv):
            raise TypeError("RequesterCategoryMatchPolicy 仅用于 requester 环境")
        valid = np.flatnonzero(obs.action_mask)
        if len(valid) == 0:
            return 0
        assert env._current is not None
        project = env.dataset.projects[env._current.project_id]

        best_a = valid[0]
        best_key = (-1, -1.0)
        for a in valid:
            wid = env._candidates[a]
            profile = env.encoder.worker_history_profile(wid, env._current.timestamp)
            if profile.dominant_category != project.category:
                continue
            key = (profile.past_count, env.dataset.get_worker_quality(wid))
            if key > best_key:
                best_key = key
                best_a = a
        return int(best_a)


class RequesterIndustryMatchPolicy:
    """请求者侧：选历史主行业与当前项目行业一致的 worker。"""

    name = "worker_industry_match"

    def select_action(self, env: EnvT, obs: Observation) -> int:
        if not isinstance(env, RequesterRecommendationEnv):
            raise TypeError("RequesterIndustryMatchPolicy 仅用于 requester 环境")
        valid = np.flatnonzero(obs.action_mask)
        if len(valid) == 0:
            return 0
        assert env._current is not None
        project = env.dataset.projects[env._current.project_id]

        best_a = valid[0]
        best_key = (-1, -1.0)
        for a in valid:
            wid = env._candidates[a]
            profile = env.encoder.worker_history_profile(wid, env._current.timestamp)
            if profile.dominant_industry_id != project.industry_id:
                continue
            key = (profile.past_count, env.dataset.get_worker_quality(wid))
            if key > best_key:
                best_key = key
                best_a = a
        return int(best_a)


WORKER_BASELINES: dict[str, type[Policy]] = {
    "random": RandomPolicy,
    "popularity": PopularityPolicy,
    "category_match": CategoryMatchPolicy,
    "industry_match": IndustryMatchPolicy,
    "award": AwardPolicy,
}

REQUESTER_BASELINES: dict[str, type[Policy]] = {
    "random": RandomPolicy,
    "worker_quality": WorkerQualityPolicy,
    "worker_activity": WorkerActivityPolicy,
    "worker_category_match": RequesterCategoryMatchPolicy,
    "worker_industry_match": RequesterIndustryMatchPolicy,
}


def make_baseline(name: str, side: str, seed: int = 0) -> Policy:
    registry = WORKER_BASELINES if side == "worker" else REQUESTER_BASELINES
    if name not in registry:
        raise KeyError(f"未知基线 {name}，可选: {list(registry.keys())}")
    cls = registry[name]
    if name == "random":
        return cls(seed=seed)  # type: ignore[call-arg]
    return cls()
