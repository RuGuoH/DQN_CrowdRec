"""
请求者侧 worker 推荐 MDP 环境。

与 worker_env 对称：项目方在收到新投稿时，从候选 worker 中推荐 1 人。
事件流：按 entry_created_at 排序的 (project, worker) 投稿记录。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from env.worker_env import Observation
from src.dataset import CrowdsourcingDataset, SplitName
from src.features import (
    PROJECT_FEAT_DIM,
    WORKER_FEAT_DIM,
    FeatureEncoder,
)


@dataclass(frozen=True)
class RequesterEvent:
    timestamp: datetime
    project_id: int
    worker_id: int
    max_revision_score: float
    winner: bool
    finalist: bool


@dataclass
class RequesterEnvConfig:
    num_candidates: int = 32
    hit_reward: float = 1.0
    miss_penalty: float = -0.1
    score_weight: float = 0.25
    quality_weight: float = 0.15
    winner_bonus: float = 0.5
    finalist_bonus: float = 0.2
    include_truth_in_candidates: bool = True
    max_steps_per_episode: int | None = None


class RequesterRecommendationEnv:
    """基于项目投稿事件流的请求者侧推荐环境。"""

    def __init__(
        self,
        dataset: CrowdsourcingDataset,
        split: SplitName = "train",
        config: RequesterEnvConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.split = split
        self.config = config or RequesterEnvConfig()
        self.rng = np.random.default_rng(seed)

        allowed = dataset.split_project_ids(split)
        events: list[RequesterEvent] = []
        for entries in dataset.entries_by_project.values():
            for e in entries:
                if e.withdrawn or e.project_id not in allowed:
                    continue
                events.append(
                    RequesterEvent(
                        timestamp=e.entry_created_at,
                        project_id=e.project_id,
                        worker_id=e.worker_id,
                        max_revision_score=e.max_revision_score,
                        winner=e.winner,
                        finalist=e.finalist,
                    )
                )
        events.sort(key=lambda x: x.timestamp)
        if not events:
            raise ValueError(f"split={split} 无请求者侧事件。")

        self.events = events
        self.encoder = FeatureEncoder(dataset=dataset, ref_time=events[0].timestamp)
        self._worker_ids = sorted(dataset.entries_by_worker.keys())
        self._idx = 0
        self._current: RequesterEvent | None = None
        self._candidates: list[int] = []

    @property
    def num_actions(self) -> int:
        return self.config.num_candidates

    @property
    def context_feat_dim(self) -> int:
        return PROJECT_FEAT_DIM

    @property
    def candidate_feat_dim(self) -> int:
        return WORKER_FEAT_DIM

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
            action < len(self._candidates)
            and self._candidates[action] == self._current.worker_id
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
        info.update(action=action, reward=reward, hit=hit, optimal_action=optimal)
        return obs, reward, terminated, False, info

    def optimal_action(self, obs: Observation) -> int | None:
        if self._current is None:
            return None
        truth = self._current.worker_id
        for i, wid in enumerate(self._candidates):
            if wid == truth and obs.action_mask[i]:
                return i
        return None

    def _active_workers_at(self, t: datetime) -> list[int]:
        active: list[int] = []
        for wid in self._worker_ids:
            times = self.encoder._worker_times.get(wid, [])
            if not times:
                continue
            # 在 t 之前有过投稿，且最近 90 天内有活动（简化为：最后投稿 <= t）
            if times[0] <= t:
                active.append(wid)
        return active

    def _build_candidates(self) -> None:
        assert self._current is not None
        ev = self._current
        k = self.config.num_candidates
        truth = ev.worker_id

        active = self._active_workers_at(ev.timestamp)
        scored = [
            (
                wid,
                self.dataset.get_worker_quality(wid),
                len(self.encoder._past_entries(wid, ev.timestamp)),
            )
            for wid in active
        ]
        scored.sort(key=lambda x: (-x[1], -x[2], x[0]))

        chosen: list[int] = []
        if self.config.include_truth_in_candidates and truth in active:
            chosen.append(truth)

        for wid, _, _ in scored:
            if len(chosen) >= k:
                break
            if wid not in chosen:
                chosen.append(wid)

        self._candidates = chosen[:k]

    def _observe(self) -> Observation:
        assert self._current is not None
        ev = self._current
        k = self.config.num_candidates
        project = self.dataset.projects[ev.project_id]

        # Observation 复用字段名：worker_feat=项目上下文，candidate_feat=worker 特征
        context_feat = self.encoder.project_context_features(project, ev.timestamp)
        cand_feat = np.zeros((k, WORKER_FEAT_DIM), dtype=np.float32)
        mask = np.zeros(k, dtype=bool)

        for i, wid in enumerate(self._candidates):
            cand_feat[i] = self.encoder.worker_features(wid, ev.timestamp)
            mask[i] = True

        return Observation(
            worker_feat=context_feat,
            candidate_feat=cand_feat,
            action_mask=mask,
        )

    def _empty_observation(self) -> Observation:
        k = self.config.num_candidates
        return Observation(
            worker_feat=np.zeros(PROJECT_FEAT_DIM, dtype=np.float32),
            candidate_feat=np.zeros((k, WORKER_FEAT_DIM), dtype=np.float32),
            action_mask=np.zeros(k, dtype=bool),
        )

    def _compute_reward(self, action: int) -> float:
        assert self._current is not None
        cfg = self.config
        if action < 0 or action >= len(self._candidates):
            return cfg.miss_penalty

        picked = self._candidates[action]
        ev = self._current
        if picked != ev.worker_id:
            return cfg.miss_penalty

        reward = cfg.hit_reward
        reward += cfg.score_weight * (ev.max_revision_score / 5.0)
        reward += cfg.quality_weight * self.dataset.get_worker_quality(ev.worker_id)
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
            "candidate_worker_ids": list(self._candidates),
        }
        if ev is not None:
            info["project_id"] = ev.project_id
            info["truth_worker_id"] = ev.worker_id
            info["timestamp"] = ev.timestamp.isoformat()
        return info
