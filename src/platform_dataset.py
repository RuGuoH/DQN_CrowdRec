"""Unified data view for the dynamic two-sided platform simulation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

from src.config import Config
from src.dataset import (
    CrowdsourcingDataset,
    EntryRecord,
    ProjectRecord,
    SplitName,
    WorkerEvent,
    build_dataset,
)


@dataclass(frozen=True)
class WorkerProjectOutcome:
    project_id: int
    worker_id: int
    submitted: bool
    entry_time: datetime | None
    max_revision_score: float
    winner: bool
    finalist: bool


@dataclass(frozen=True)
class PlatformWorkerEvent:
    """Worker becomes available for task recommendation."""

    timestamp: datetime
    worker_id: int
    truth_project_id: int | None
    source_project_id: int | None = None
    synthetic: bool = False


class PlatformDataset:
    """Standardized indexes used by the dynamic platform environment."""

    def __init__(self, dataset: CrowdsourcingDataset, split: SplitName) -> None:
        self.dataset = dataset
        self.split = split
        self.project_ids = dataset.split_project_ids(split)
        self.projects: dict[int, ProjectRecord] = {
            pid: p for pid, p in dataset.projects.items() if pid in self.project_ids
        }
        self.worker_events: list[PlatformWorkerEvent] = self._build_worker_events()
        self.outcomes: dict[tuple[int, int], WorkerProjectOutcome] = (
            self._build_outcomes()
        )
        self.entries_by_project: dict[int, list[EntryRecord]] = {
            pid: list(entries)
            for pid, entries in dataset.entries_by_project.items()
            if pid in self.project_ids
        }

    def iter_worker_events(self) -> Iterator[PlatformWorkerEvent]:
        yield from self.worker_events

    def outcome_for(
        self,
        project_id: int,
        worker_id: int,
    ) -> WorkerProjectOutcome:
        outcome = self.outcomes.get((project_id, worker_id))
        if outcome is not None:
            return outcome
        return WorkerProjectOutcome(
            project_id=project_id,
            worker_id=worker_id,
            submitted=False,
            entry_time=None,
            max_revision_score=0.0,
            winner=False,
            finalist=False,
        )

    def historical_winner(self, project_id: int) -> int | None:
        for (pid, _), outcome in self.outcomes.items():
            if pid == project_id and outcome.winner:
                return outcome.worker_id
        return None

    def _build_worker_events(self) -> list[PlatformWorkerEvent]:
        events: list[PlatformWorkerEvent] = []
        for ev in self.dataset.iter_worker_events(self.split):
            events.append(
                PlatformWorkerEvent(
                    timestamp=ev.timestamp,
                    worker_id=ev.worker_id,
                    truth_project_id=ev.project_id,
                )
            )
        events.sort(key=lambda e: (e.timestamp, e.worker_id, e.truth_project_id or -1))
        return events

    def _build_outcomes(self) -> dict[tuple[int, int], WorkerProjectOutcome]:
        outcomes: dict[tuple[int, int], WorkerProjectOutcome] = {}
        for pid, entries in self.dataset.entries_by_project.items():
            if pid not in self.project_ids:
                continue
            for entry in entries:
                if entry.withdrawn:
                    continue
                key = (pid, entry.worker_id)
                prev = outcomes.get(key)
                if prev is None or entry.max_revision_score > prev.max_revision_score:
                    outcomes[key] = WorkerProjectOutcome(
                        project_id=pid,
                        worker_id=entry.worker_id,
                        submitted=True,
                        entry_time=entry.entry_created_at,
                        max_revision_score=entry.max_revision_score,
                        winner=entry.winner,
                        finalist=entry.finalist,
                    )
                elif prev is not None and (entry.winner or entry.finalist):
                    outcomes[key] = WorkerProjectOutcome(
                        project_id=pid,
                        worker_id=entry.worker_id,
                        submitted=True,
                        entry_time=min(prev.entry_time, entry.entry_created_at)
                        if prev.entry_time
                        else entry.entry_created_at,
                        max_revision_score=max(
                            prev.max_revision_score,
                            entry.max_revision_score,
                        ),
                        winner=prev.winner or entry.winner,
                        finalist=prev.finalist or entry.finalist,
                    )
        return outcomes

    def summary(self) -> dict:
        return {
            "split": self.split,
            "projects": len(self.projects),
            "worker_events": len(self.worker_events),
            "outcomes": len(self.outcomes),
            **self.dataset.summary(),
        }


def build_platform_dataset(
    config: Config | None = None,
    *,
    split: SplitName = "train",
    use_cache: bool = True,
    force_reload: bool = False,
) -> PlatformDataset:
    return PlatformDataset(
        build_dataset(config, use_cache=use_cache, force_reload=force_reload),
        split,
    )
