"""Worker / Project 特征编码（供 MDP 与 Q 网络使用）。"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from src.dataset import CrowdsourcingDataset, ProjectRecord

WORKER_FEAT_DIM = 12
PROJECT_FEAT_DIM = 13


@dataclass(frozen=True)
class WorkerHistoryProfile:
    """Worker 在时刻 t 之前的历史画像，避免把未来投稿泄漏进特征。"""

    past_count: int
    mean_score: float
    win_rate: float
    finalist_rate: float
    gap_hours: float
    recent_30d_count: int
    dominant_category: int | None
    dominant_category_share: float
    dominant_industry_id: int | None
    dominant_industry_share: float


@dataclass
class FeatureEncoder:
    """将 worker 与候选项目编码为固定长度向量。"""

    dataset: CrowdsourcingDataset
    ref_time: datetime
    _worker_times: dict[int, list[datetime]] = field(default_factory=dict, init=False)
    _profile_cache: dict[tuple[int, datetime], WorkerHistoryProfile] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        for wid, entries in self.dataset.entries_by_worker.items():
            self._worker_times[wid] = [e.entry_created_at for e in entries]

    def _past_entries(self, worker_id: int, t: datetime) -> list:
        history = self.dataset.entries_by_worker.get(worker_id, [])
        if not history:
            return []
        times = self._worker_times.get(worker_id, [])
        idx = bisect.bisect_left(times, t)
        return history[:idx]

    @staticmethod
    def _dominant_from_counts(counts: dict[int, int]) -> tuple[int | None, float]:
        if not counts:
            return None, 0.0
        total = sum(counts.values())
        value, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        return value, count / total

    def worker_history_profile(self, worker_id: int, t: datetime) -> WorkerHistoryProfile:
        cache_key = (worker_id, t)
        if cache_key in self._profile_cache:
            return self._profile_cache[cache_key]

        past = self._past_entries(worker_id, t)
        cat_counts: dict[int, int] = {}
        industry_counts: dict[int, int] = {}
        for e in past:
            project = self.dataset.projects.get(e.project_id)
            if project is None:
                continue
            cat_counts[project.category] = cat_counts.get(project.category, 0) + 1
            industry_counts[project.industry_id] = industry_counts.get(project.industry_id, 0) + 1

        dom_cat, dom_cat_share = self._dominant_from_counts(cat_counts)
        dom_industry, dom_industry_share = self._dominant_from_counts(industry_counts)

        n = len(past)
        if n == 0:
            profile = WorkerHistoryProfile(
                past_count=0,
                mean_score=0.0,
                win_rate=0.0,
                finalist_rate=0.0,
                gap_hours=0.0,
                recent_30d_count=0,
                dominant_category=dom_cat,
                dominant_category_share=dom_cat_share,
                dominant_industry_id=dom_industry,
                dominant_industry_share=dom_industry_share,
            )
            self._profile_cache[cache_key] = profile
            return profile

        recent_30d = sum(
            1 for e in past if (t - e.entry_created_at).total_seconds() <= 30 * 86400
        )
        profile = WorkerHistoryProfile(
            past_count=n,
            mean_score=float(np.mean([e.max_revision_score for e in past])),
            win_rate=sum(1 for e in past if e.winner) / n,
            finalist_rate=sum(1 for e in past if e.finalist) / n,
            gap_hours=(t - past[-1].entry_created_at).total_seconds() / 3600.0,
            recent_30d_count=recent_30d,
            dominant_category=dom_cat,
            dominant_category_share=dom_cat_share,
            dominant_industry_id=dom_industry,
            dominant_industry_share=dom_industry_share,
        )
        self._profile_cache[cache_key] = profile
        return profile

    def worker_features(self, worker_id: int, t: datetime) -> np.ndarray:
        q = self.dataset.get_worker_quality(worker_id)
        profile = self.worker_history_profile(worker_id, t)
        n = profile.past_count
        if n == 0:
            return np.array(
                [q, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )

        dom_cat = profile.dominant_category or 0
        dom_industry = profile.dominant_industry_id or 0

        return np.array(
            [
                q,
                np.log1p(n),
                profile.mean_score / 5.0,
                profile.win_rate,
                profile.finalist_rate,
                dom_cat / 20.0,
                dom_industry / max(len(self.dataset.industry_vocab), 1),
                profile.dominant_category_share,
                profile.dominant_industry_share,
                np.log1p(profile.gap_hours) / 10.0,
                np.log1p(profile.recent_30d_count) / 5.0,
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
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

    def project_features(
        self,
        project: ProjectRecord,
        t: datetime,
        worker_id: int,
        profile: WorkerHistoryProfile | None = None,
    ) -> np.ndarray:
        profile = profile or self.worker_history_profile(worker_id, t)
        dom_cat = profile.dominant_category
        dom_industry = profile.dominant_industry_id
        cat_match = 1.0 if dom_cat is not None and project.category == dom_cat else 0.0
        industry_match = (
            1.0 if dom_industry is not None and project.industry_id == dom_industry else 0.0
        )

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
                industry_match,
                profile.dominant_category_share,
                profile.dominant_industry_share,
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
        profile = self.worker_history_profile(worker_id, t)
        return np.stack(
            [self.project_features(p, t, worker_id, profile) for p in candidates],
            axis=0,
        )
