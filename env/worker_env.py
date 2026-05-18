"""
参与者侧任务推荐 MDP 环境。

MDP 定义
--------
- 状态: worker 特征 + Top-K 候选项目特征 + action_mask
- 动作: 从 K 个候选槽位中选择 1 个推荐项目（离散 0..K-1）
- 奖励: 命中真实投稿项目 + 质量/获奖加成；未命中给小惩罚
- 转移: 推进到下一个 worker 事件（按 iter_worker_events 时间序）
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from src.dataset import CrowdsourcingDataset, ProjectRecord, SplitName, WorkerEvent
from src.features import (
    PROJECT_FEAT_DIM,
    WORKER_FEAT_DIM,
    FeatureEncoder,
)


@dataclass
class EnvConfig:
    num_candidates: int = 32
    hit_reward: float = 1.0
    miss_penalty: float = -0.1
    score_weight: float = 0.2
    winner_bonus: float = 0.5
    finalist_bonus: float = 0.2
    include_truth_in_candidates: bool = True
    max_steps_per_episode: int | None = None  # 调试时可限制步数


@dataclass
class Observation:
    """环境观测（可直接喂给 DQN）。"""

    worker_feat: np.ndarray  # (WORKER_FEAT_DIM,)
    candidate_feat: np.ndarray  # (K, PROJECT_FEAT_DIM)
    action_mask: np.ndarray  # (K,) bool

    def to_dict(self) -> dict[str, np.ndarray]:
        return {
            "worker_feat": self.worker_feat,
            "candidate_feat": self.candidate_feat,
            "action_mask": self.action_mask,
        }

    @staticmethod
    def from_dict(d: dict[str, np.ndarray]) -> "Observation":
        return Observation(
            worker_feat=d["worker_feat"],
            candidate_feat=d["candidate_feat"],
            action_mask=d["action_mask"],
        )


class WorkerRecommendationEnv:
    """基于 iter_worker_events 的离线推荐环境。"""

    def __init__(
        self,
        dataset: CrowdsourcingDataset,
        split: SplitName = "train",
        config: EnvConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.split = split
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(seed)

        self.events: list[WorkerEvent] = list(dataset.iter_worker_events(split))
        if not self.events:
            raise ValueError(f"split={split} 无 worker 事件，请检查数据或划分。")

        self.encoder = FeatureEncoder(
            dataset=dataset,
            ref_time=self.events[0].timestamp,
        )
        self._active_projects: list[ProjectRecord] = sorted(
            dataset.projects.values(), key=lambda p: p.start_date
        )
        self._idx = 0
        self._current: WorkerEvent | None = None
        self._candidates: list[ProjectRecord] = []
        self._candidate_ids: list[int] = []

    @property
    def num_actions(self) -> int:
        return self.config.num_candidates

    @property
    def worker_feat_dim(self) -> int:
        return WORKER_FEAT_DIM

    @property
    def project_feat_dim(self) -> int:
        return PROJECT_FEAT_DIM

    def reset(self, *, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._idx = 0
        self._current = self.events[0]
        self._build_candidates()
        return self._observe(), self._info()

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        if self._current is None:
            raise RuntimeError("请先调用 reset()")

        pre_obs = self._observe()
        optimal = self.optimal_action(pre_obs)
        reward = self._compute_reward(action)
        hit = (
            action < len(self._candidate_ids)
            and self._candidate_ids[action] == self._current.project_id
        )

        self._idx += 1
        terminated = self._idx >= len(self.events)
        if terminated:
            self._current = None
            obs = self._empty_observation()
        else:
            self._current = self.events[self._idx]
            self._build_candidates()
            obs = self._observe()

        info = self._info()
        info.update(
            action=action,
            reward=reward,
            hit=hit,
            optimal_action=optimal,
        )
        return obs, reward, terminated, False, info

    def active_projects_at(self, t: datetime) -> list[ProjectRecord]:
        """返回时刻 t 仍开放投稿的项目（按开始时间排序列表过滤）。"""
        return [
            p
            for p in self._active_projects
            if p.start_date <= t < p.deadline
        ]

    def sample_action(self, obs: Observation) -> int:
        valid = np.flatnonzero(obs.action_mask)
        if len(valid) == 0:
            return 0
        return int(self.rng.choice(valid))

    def optimal_action(self, obs: Observation) -> int | None:
        """返回真实投稿项目对应的动作索引（用于模仿学习 / 评估）。"""
        if self._current is None:
            return None
        truth = self._current.project_id
        for i, pid in enumerate(self._candidate_ids):
            if pid == truth and obs.action_mask[i]:
                return i
        return None

    # ------------------------------------------------------------------ 内部
    def _build_candidates(self) -> None:
        assert self._current is not None
        ev = self._current
        k = self.config.num_candidates
        active = self.active_projects_at(ev.timestamp)
        active_by_id = {p.project_id: p for p in active}

        chosen: list[ProjectRecord] = []
        chosen_ids: set[int] = set()

        truth = ev.project_id
        if self.config.include_truth_in_candidates and truth in active_by_id:
            p = active_by_id[truth]
            chosen.append(p)
            chosen_ids.add(truth)

        pool = [p for pid, p in active_by_id.items() if pid not in chosen_ids]
        pool.sort(key=lambda p: (-p.entry_count, -p.total_awards, p.project_id))
        for p in pool:
            if len(chosen) >= k:
                break
            chosen.append(p)
            chosen_ids.add(p.project_id)

        self._candidates = chosen[:k]
        self._candidate_ids = [p.project_id for p in self._candidates]

    def _observe(self) -> Observation:
        assert self._current is not None
        ev = self._current
        k = self.config.num_candidates

        worker_feat = self.encoder.worker_features(ev.worker_id, ev.timestamp)
        cand_feat = np.zeros((k, PROJECT_FEAT_DIM), dtype=np.float32)
        mask = np.zeros(k, dtype=bool)

        encoded = self.encoder.encode_candidates(
            ev.worker_id, ev.timestamp, self._candidates
        )
        n = min(len(self._candidates), k)
        if n > 0:
            cand_feat[:n] = encoded[:n]
            mask[:n] = True

        return Observation(
            worker_feat=worker_feat,
            candidate_feat=cand_feat,
            action_mask=mask,
        )

    def _empty_observation(self) -> Observation:
        k = self.config.num_candidates
        return Observation(
            worker_feat=np.zeros(WORKER_FEAT_DIM, dtype=np.float32),
            candidate_feat=np.zeros((k, PROJECT_FEAT_DIM), dtype=np.float32),
            action_mask=np.zeros(k, dtype=bool),
        )

    def _compute_reward(self, action: int) -> float:
        assert self._current is not None
        cfg = self.config
        if action < 0 or action >= len(self._candidate_ids):
            return cfg.miss_penalty

        if not self._candidates or action >= len(self._candidates):
            return cfg.miss_penalty

        picked_id = self._candidate_ids[action]
        truth = self._current.project_id
        ev = self._current

        if picked_id != truth:
            return cfg.miss_penalty

        reward = cfg.hit_reward
        reward += cfg.score_weight * (ev.max_revision_score / 5.0)
        if ev.winner:
            reward += cfg.winner_bonus
        if ev.finalist:
            reward += cfg.finalist_bonus
        return reward

    def _info(self) -> dict[str, Any]:
        ev = self._current
        info: dict[str, Any] = {
            "event_idx": self._idx,
            "num_events": len(self.events),
            "candidate_ids": list(self._candidate_ids),
        }
        if ev is not None:
            info["worker_id"] = ev.worker_id
            info["truth_project_id"] = ev.project_id
            info["timestamp"] = ev.timestamp.isoformat()
        return info
