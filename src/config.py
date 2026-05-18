from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    data_dir: Path
    min_start_date: str
    page_limit: int
    train_ratio: float
    val_ratio: float
    max_projects: int | None
    cache_dir: Path

    @property
    def test_ratio(self) -> float:
        return 1.0 - self.train_ratio - self.val_ratio


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    data_dir = ROOT / raw["data_dir"]
    cache_dir = ROOT / raw.get("cache_dir", "cache")
    return Config(
        data_dir=data_dir,
        min_start_date=raw["min_start_date"],
        page_limit=int(raw["page_limit"]),
        train_ratio=float(raw["train_ratio"]),
        val_ratio=float(raw["val_ratio"]),
        max_projects=raw.get("max_projects"),
        cache_dir=cache_dir,
    )
