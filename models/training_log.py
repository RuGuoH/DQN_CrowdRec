"""训练日志与 checkpoint 管理。"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class EpisodeMetrics:
    episode: int
    split: str
    reward: float
    hit_rate: float
    steps: int
    epsilon: float
    avg_loss: float | None = None
    buffer_size: int = 0
    global_step: int = 0


@dataclass
class TrainingLogger:
    """将训练指标写入 CSV / JSON，并管理 checkpoint 目录。"""

    log_dir: Path
    run_name: str = "run"
    _rows: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.log_dir = Path(self.log_dir)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / f"{self.run_name}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)
        self.metrics_csv = self.run_dir / "metrics.csv"
        self.config_json = self.run_dir / "config.json"

    def save_config(self, config: dict[str, Any]) -> None:
        with self.config_json.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def log_episode(self, metrics: EpisodeMetrics) -> None:
        row = asdict(metrics)
        row["time"] = datetime.now().isoformat()
        self._rows.append(row)
        self._append_csv(row)
        print(self._format_row(row), flush=True)

    def _format_row(self, row: dict[str, Any]) -> str:
        loss = row.get("avg_loss")
        loss_s = f"{loss:.4f}" if loss is not None else "n/a"
        return (
            f"[{row['split']}] ep={row['episode']:04d} "
            f"reward={row['reward']:.2f} hit={row['hit_rate']:.3f} "
            f"steps={row['steps']} eps={row['epsilon']:.3f} "
            f"loss={loss_s} buf={row['buffer_size']} step={row['global_step']}"
        )

    def _append_csv(self, row: dict[str, Any]) -> None:
        write_header = not self.metrics_csv.exists()
        with self.metrics_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def save_summary(self) -> Path:
        path = self.run_dir / "summary.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(self._rows, f, ensure_ascii=False, indent=2)
        return path

    def checkpoint_path(self, tag: str) -> Path:
        return self.ckpt_dir / f"{tag}.pt"
