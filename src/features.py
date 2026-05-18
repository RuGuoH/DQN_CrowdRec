"""Worker / Project 特征编码（供 MDP 与 Q 网络使用）。"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from src.dataset import CrowdsourcingDataset, ProjectRecord

WORKER_FEAT_DIM = 8
PROJECT_FEAT_DIM = 10


@dataclass
class FeatureEncoder:
    """将 worker 与候选项目编码为固定长度向量。"""

    dataset: CrowdsourcingDataset
    ref_time: datetime
    _worker_times: dict[int, list[datetime]] = field(default_factory=dict, init=False)
    _worker_dom_cat: dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for wid, entries in self.dataset.entries_by_worker.items():
            self._worker_times[wid] = [e.entry_created_at for e in entries]
            cats = [
                self.dataset.projects[e.project_id].category
                for e in entries
                if e.project_id in self.dataset.projects
            ]
            self._worker_dom_cat[wid] = max(set(cats), key=cats.count) if cats else 0

    def _past_entries(self, worker_id: int, t: datetime) -> list:
        history = self.dataset.entries_by_worker.get(worker_id, [])
        if not history:
            return []
        times = self._worker_times.get(worker_id, [])
        idx = bisect.bisect_left(times, t)
        return history[:idx]

    def worker_features(self, worker_id: int, t: datetime) -> np.ndarray:
        q = self.dataset.get_worker_quality(worker_id)
        past = self._past_entries(worker_id, t)
        n = len(past)
        if n == 0:
            return np.array(
                [q, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )

        scores = [e.max_revision_score for e in past]
        wins = sum(1 for e in past if e.winner)
        finalists = sum(1 for e in past if e.finalist)
        dom_cat = self._worker_dom_cat.get(worker_id, 0)
        last_t = past[-1].entry_created_at
        gap_h = (t - last_t).total_seconds() / 3600.0

        return np.array(
            [
                q,
                np.log1p(n),
                float(np.mean(scores)),
                wins / n,
                finalists / n,
                dom_cat / 20.0,
                np.log1p(gap_h),
                np.log1p(n) / 10.0,
            ],
            dtype=np.float32,
        )

    def project_context_features(
        self, project: ProjectRecord, t: datetime
    ) -> np.ndarray:
        """请求者侧：仅项目上下文，不依赖特定 worker。"""
        hours_left = max((project.deadline - t).total_seconds() / 3600.0, 0.0)
        hours_open = max((t - project.start_date).total_seconds() / 3600.0, 0.0)
        return np.array(
            [
                project.category / 20.0,
                project.sub_category / 60.0,
                project.industry_id / max(len(self.dataset.industry_vocab), 1),
                np.log1p(project.entry_count),
                np.log1p(max(project.total_awards, 0.0)),
                project.average_score / 5.0,
                float(project.featured),
                np.log1p(hours_left) / 10.0,
                np.log1p(hours_open) / 10.0,
                0.0,
            ],
            dtype=np.float32,
        )

    def project_features(
        self,
        project: ProjectRecord,
        t: datetime,
        worker_id: int,
    ) -> np.ndarray:
        dom_cat = self._worker_dom_cat.get(worker_id, 0)
        cat_match = 1.0 if project.category == dom_cat else 0.0

        hours_left = max((project.deadline - t).total_seconds() / 3600.0, 0.0)
        hours_open = max((t - project.start_date).total_seconds() / 3600.0, 0.0)

        return np.array(
            [
                project.category / 20.0,
                project.sub_category / 60.0,
                project.industry_id / max(len(self.dataset.industry_vocab), 1),
                np.log1p(project.entry_count),
                np.log1p(max(project.total_awards, 0.0)),
                project.average_score / 5.0,
                float(project.featured),
                np.log1p(hours_left) / 10.0,
                np.log1p(hours_open) / 10.0,
                cat_match,
            ],
            dtype=np.float32,
        )

    def encode_candidates(
        self,
        worker_id: int,
        t: datetime,
        candidates: list[ProjectRecord],
    ) -> np.ndarray:
        if not candidates:
            return np.zeros((0, PROJECT_FEAT_DIM), dtype=np.float32)
        return np.stack(
            [self.project_features(p, t, worker_id) for p in candidates],
            axis=0,
        )
