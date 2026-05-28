"""Convenience exports for model modules.

Heavy PyTorch-backed objects are imported lazily so heuristic-only scripts can
run in lightweight environments.
"""

from __future__ import annotations

__all__ = [
    "DQNAgent",
    "DQNConfig",
    "DuelingQNetwork",
    "QNetwork",
    "Transition",
    "TrainingLogger",
    "EpisodeMetrics",
    "save_best_checkpoint",
    "WORKER_BASELINES",
    "REQUESTER_BASELINES",
    "make_baseline",
    "evaluate_one",
    "make_platform_selectors",
    "run_platform_episode",
    "run_platform_eval",
]


def __getattr__(name: str):
    if name in {
        "DQNAgent",
        "DQNConfig",
        "DuelingQNetwork",
        "QNetwork",
        "Transition",
        "save_best_checkpoint",
    }:
        from models import dqn

        return getattr(dqn, name)
    if name in {"TrainingLogger", "EpisodeMetrics"}:
        from models import training_log

        return getattr(training_log, name)
    if name in {"WORKER_BASELINES", "REQUESTER_BASELINES", "make_baseline"}:
        from models import baselines

        return getattr(baselines, name)
    if name == "evaluate_one":
        from models import eval_runner

        return eval_runner.evaluate_one
    if name == "make_platform_selectors":
        from models import platform_baselines

        return platform_baselines.make_platform_selectors
    if name in {"run_platform_episode", "run_platform_eval"}:
        from models import platform_training

        return getattr(platform_training, name)
    raise AttributeError(name)
