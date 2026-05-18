"""
Crowdspring 众包数据加载与事件流构建。

用法:
    python -m src.dataset
    python -m src.dataset --max-projects 100
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal

from dateutil.parser import parse as parse_dt

from src.config import Config, load_config

SplitName = Literal["train", "val", "test"]


def _dt_to_str(t: datetime) -> str:
    return t.isoformat()


def _project_to_dict(p: "ProjectRecord") -> dict:
    d = asdict(p)
    d["start_date"] = _dt_to_str(p.start_date)
    d["deadline"] = _dt_to_str(p.deadline)
    return d


def _project_from_dict(d: dict) -> "ProjectRecord":
    return ProjectRecord(
        project_id=d["project_id"],
        category=d["category"],
        sub_category=d["sub_category"],
        industry_id=d["industry_id"],
        entry_count=d["entry_count"],
        start_date=parse_dt(d["start_date"]),
        deadline=parse_dt(d["deadline"]),
        total_awards=d["total_awards"],
        average_score=d["average_score"],
        featured=d["featured"],
        split=d.get("split"),
    )


def _entry_to_dict(e: "EntryRecord") -> dict:
    d = asdict(e)
    d["entry_created_at"] = _dt_to_str(e.entry_created_at)
    return d


def _entry_from_dict(d: dict) -> "EntryRecord":
    return EntryRecord(
        project_id=d["project_id"],
        entry_number=d["entry_number"],
        worker_id=d["worker_id"],
        entry_created_at=parse_dt(d["entry_created_at"]),
        winner=d["winner"],
        finalist=d["finalist"],
        withdrawn=d["withdrawn"],
        max_revision_score=d["max_revision_score"],
    )


@dataclass(frozen=True)
class ProjectRecord:
    project_id: int
    category: int
    sub_category: int
    industry_id: int
    entry_count: int
    start_date: datetime
    deadline: datetime
    total_awards: float
    average_score: float
    featured: bool
    split: SplitName | None = None


@dataclass(frozen=True)
class EntryRecord:
    project_id: int
    entry_number: int
    worker_id: int
    entry_created_at: datetime
    winner: bool
    finalist: bool
    withdrawn: bool
    max_revision_score: float


@dataclass(frozen=True)
class WorkerEvent:
    """按时间排序的 worker 到达/投稿事件（用于参与者侧 MDP）。"""

    worker_id: int
    timestamp: datetime
    project_id: int
    entry_number: int
    max_revision_score: float
    winner: bool
    finalist: bool


class CrowdsourcingDataset:
    """加载 project / entry / worker_quality，并构建时间划分与事件流。"""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()
        self.data_dir = self.config.data_dir
        self.page_limit = self.config.page_limit
        self.min_start_date = parse_dt(self.config.min_start_date)

        self.worker_quality: dict[int, float] = {}
        self.industry_vocab: dict[str, int] = {}
        self.projects: dict[int, ProjectRecord] = {}
        self.entries_by_project: dict[int, list[EntryRecord]] = {}
        self.entries_by_worker: dict[int, list[EntryRecord]] = {}

        self._worker_events: list[WorkerEvent] | None = None
        self._split_project_ids: dict[SplitName, set[int]] | None = None

    # ------------------------------------------------------------------ IO
    def load(self, use_cache: bool = True, force_reload: bool = False) -> "CrowdsourcingDataset":
        cache_path = self._cache_path()
        if use_cache and not force_reload and cache_path.exists():
            try:
                self._load_cache(cache_path)
                return self
            except Exception:
                pass  # 缓存损坏或版本不兼容时回退到全量解析

        self._load_worker_quality()
        project_ids = self._load_project_list()
        if self.config.max_projects is not None:
            project_ids = project_ids[: self.config.max_projects]

        for pid in project_ids:
            project = self._load_project(pid)
            if project is None:
                continue
            self.projects[pid] = project
            entries = self._load_entries(pid)
            if entries:
                self.entries_by_project[pid] = entries

        self._build_worker_index()
        self._assign_splits()
        self._worker_events = None

        if use_cache:
            self._save_cache(cache_path)
        return self

    def _cache_path(self) -> Path:
        tag = "all" if self.config.max_projects is None else f"n{self.config.max_projects}"
        return self.config.cache_dir / f"dataset_{tag}.pkl"

    def _save_cache(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "worker_quality": self.worker_quality,
            "industry_vocab": self.industry_vocab,
            "projects": {pid: _project_to_dict(p) for pid, p in self.projects.items()},
            "entries_by_project": {
                pid: [_entry_to_dict(e) for e in entries]
                for pid, entries in self.entries_by_project.items()
            },
            "entries_by_worker": {
                wid: [_entry_to_dict(e) for e in entries]
                for wid, entries in self.entries_by_worker.items()
            },
            "split_project_ids": self._split_project_ids,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_cache(self, path: Path) -> None:
        with path.open("rb") as f:
            payload = pickle.load(f)

        if payload.get("version") != 2:
            raise ValueError("cache version mismatch")

        self.worker_quality = payload["worker_quality"]
        self.industry_vocab = payload["industry_vocab"]
        self.projects = {
            int(pid): _project_from_dict(d) for pid, d in payload["projects"].items()
        }
        self.entries_by_project = {
            int(pid): [_entry_from_dict(d) for d in entries]
            for pid, entries in payload["entries_by_project"].items()
        }
        self.entries_by_worker = {
            int(wid): [_entry_from_dict(d) for d in entries]
            for wid, entries in payload["entries_by_worker"].items()
        }
        self._split_project_ids = payload["split_project_ids"]
        self._worker_events = None

    def _load_worker_quality(self) -> None:
        path = self.data_dir / "worker_quality.csv"
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0] == "worker_id":
                    continue
                wid, q = int(row[0]), float(row[1])
                if q > 0.0:
                    self.worker_quality[wid] = q / 100.0

    def _load_project_list(self) -> list[int]:
        path = self.data_dir / "project_list.csv"
        ids: list[int] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pid, _ = line.split(",", 1)
                ids.append(int(pid))
        return ids

    def _load_project(self, project_id: int) -> ProjectRecord | None:
        path = self.data_dir / "project" / f"project_{project_id}.txt"
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as f:
            text = json.load(f)

        start_date = parse_dt(text["start_date"])
        if start_date < self.min_start_date:
            return None

        industry = text.get("industry")
        if industry not in self.industry_vocab:
            self.industry_vocab[industry] = len(self.industry_vocab)

        return ProjectRecord(
            project_id=project_id,
            category=int(text["category"]),
            sub_category=int(text["sub_category"]),
            industry_id=self.industry_vocab[industry],
            entry_count=int(text.get("entry_count", 0)),
            start_date=start_date,
            deadline=parse_dt(text["deadline"]),
            total_awards=float(text.get("total_awards") or 0.0),
            average_score=float(text.get("average_score") or 0.0),
            featured=bool(text.get("featured", False)),
        )

    def _load_entries(self, project_id: int) -> list[EntryRecord]:
        project = self.projects[project_id]
        entry_dir = self.data_dir / "entry"
        records: list[EntryRecord] = []

        for offset in range(0, max(project.entry_count, 1), self.page_limit):
            path = entry_dir / f"entry_{project_id}_{offset}.txt"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as f:
                page = json.load(f)
            for item in page.get("results", []):
                worker_id = item.get("author") or item.get("worker")
                if worker_id is None:
                    continue
                revisions = item.get("revisions") or []
                scores = [float(r.get("score") or 0.0) for r in revisions]
                records.append(
                    EntryRecord(
                        project_id=project_id,
                        entry_number=int(item["entry_number"]),
                        worker_id=int(worker_id),
                        entry_created_at=parse_dt(item["entry_created_at"]),
                        winner=bool(item.get("winner", False)),
                        finalist=bool(item.get("finalist", False)),
                        withdrawn=bool(item.get("withdrawn", False)),
                        max_revision_score=max(scores) if scores else 0.0,
                    )
                )
        return records

    def _build_worker_index(self) -> None:
        by_worker: dict[int, list[EntryRecord]] = {}
        for entries in self.entries_by_project.values():
            for e in entries:
                by_worker.setdefault(e.worker_id, []).append(e)
        for lst in by_worker.values():
            lst.sort(key=lambda x: x.entry_created_at)
        self.entries_by_worker = by_worker

    def _assign_splits(self) -> None:
        ordered = sorted(self.projects.values(), key=lambda p: p.start_date)
        n = len(ordered)
        n_train = int(n * self.config.train_ratio)
        n_val = int(n * self.config.val_ratio)

        train_ids = {p.project_id for p in ordered[:n_train]}
        val_ids = {p.project_id for p in ordered[n_train : n_train + n_val]}
        test_ids = {p.project_id for p in ordered[n_train + n_val :]}

        updated: dict[int, ProjectRecord] = {}
        for p in ordered:
            if p.project_id in train_ids:
                split: SplitName = "train"
            elif p.project_id in val_ids:
                split = "val"
            else:
                split = "test"
            updated[p.project_id] = ProjectRecord(**{**asdict(p), "split": split})
        self.projects = updated
        self._split_project_ids = {"train": train_ids, "val": val_ids, "test": test_ids}

    # -------------------------------------------------------------- 查询 API
    def split_project_ids(self, split: SplitName) -> set[int]:
        if self._split_project_ids is None:
            raise RuntimeError("请先调用 load()")
        return self._split_project_ids[split]

    def get_worker_quality(self, worker_id: int, default: float = 0.5) -> float:
        return self.worker_quality.get(worker_id, default)

    def active_projects_at(self, t: datetime) -> list[ProjectRecord]:
        """返回时刻 t 仍开放投稿的项目。"""
        return [
            p
            for p in self.projects.values()
            if p.start_date <= t < p.deadline
        ]

    def iter_worker_events(self, split: SplitName | None = None) -> Iterator[WorkerEvent]:
        """按 entry_created_at 排序的 worker 事件流。"""
        if self._worker_events is None:
            events: list[WorkerEvent] = []
            for entries in self.entries_by_project.values():
                for e in entries:
                    if e.withdrawn:
                        continue
                    events.append(
                        WorkerEvent(
                            worker_id=e.worker_id,
                            timestamp=e.entry_created_at,
                            project_id=e.project_id,
                            entry_number=e.entry_number,
                            max_revision_score=e.max_revision_score,
                            winner=e.winner,
                            finalist=e.finalist,
                        )
                    )
            events.sort(key=lambda x: x.timestamp)
            self._worker_events = events

        allowed = self.split_project_ids(split) if split else None
        for ev in self._worker_events:
            if allowed is None or ev.project_id in allowed:
                yield ev

    def summary(self) -> dict:
        n_entries = sum(len(v) for v in self.entries_by_project.values())
        return {
            "projects": len(self.projects),
            "entries": n_entries,
            "workers_with_quality": len(self.worker_quality),
            "workers_with_entries": len(self.entries_by_worker),
            "industries": len(self.industry_vocab),
            "train_projects": len(self.split_project_ids("train")),
            "val_projects": len(self.split_project_ids("val")),
            "test_projects": len(self.split_project_ids("test")),
        }


def build_dataset(
    config: Config | None = None,
    *,
    use_cache: bool = True,
    force_reload: bool = False,
) -> CrowdsourcingDataset:
    ds = CrowdsourcingDataset(config)
    return ds.load(use_cache=use_cache, force_reload=force_reload)


def _demo(max_projects: int | None) -> None:
    cfg = load_config()
    if max_projects is not None:
        cfg = Config(
            data_dir=cfg.data_dir,
            min_start_date=cfg.min_start_date,
            page_limit=cfg.page_limit,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            max_projects=max_projects,
            cache_dir=cfg.cache_dir,
        )

    print(f"数据目录: {cfg.data_dir}")
    ds = build_dataset(cfg, use_cache=True)
    print("数据集统计:", ds.summary())

    print("\n前 5 条 worker 事件 (train):")
    for i, ev in enumerate(ds.iter_worker_events("train")):
        if i >= 5:
            break
        q = ds.get_worker_quality(ev.worker_id)
        print(
            f"  t={ev.timestamp.isoformat()} worker={ev.worker_id} "
            f"quality={q:.2f} -> project={ev.project_id} score={ev.max_revision_score}"
        )

    sample_t = next(ds.iter_worker_events("train")).timestamp
    active = ds.active_projects_at(sample_t)
    print(f"\n示例时刻 {sample_t.isoformat()} 开放项目数: {len(active)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="加载并检查众包数据集")
    parser.add_argument(
        "--max-projects",
        type=int,
        default=50,
        help="最多加载项目数（默认 50，全量请传 0）",
    )
    parser.add_argument("--no-cache", action="store_true", help="禁用 pickle 缓存")
    parser.add_argument("--force-reload", action="store_true", help="忽略缓存强制重载")
    args = parser.parse_args()

    max_p = None if args.max_projects == 0 else args.max_projects
    cfg = load_config()
    if max_p is not None:
        cfg = Config(
            data_dir=cfg.data_dir,
            min_start_date=cfg.min_start_date,
            page_limit=cfg.page_limit,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            max_projects=max_p,
            cache_dir=cfg.cache_dir,
        )
    ds = build_dataset(cfg, use_cache=not args.no_cache, force_reload=args.force_reload)
    print(ds.summary())


if __name__ == "__main__":
    main()
