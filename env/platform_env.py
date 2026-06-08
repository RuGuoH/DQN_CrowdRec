"""Dynamic two-sided platform simulation for crowdsourcing recommendation."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np

from env.worker_env import Observation
from src.dataset import ProjectRecord
from src.features import (
    PLATFORM_PROJECT_FEAT_DIM,
    REQUESTER_CONTEXT_FEAT_DIM,
    WORKER_FEAT_DIM,
    FeatureEncoder,
)
from src.platform_dataset import PlatformDataset, PlatformWorkerEvent

ActorName = Literal["worker", "requester"]
RewardMode = Literal["utility", "legacy"]


@dataclass
class PlatformEnvConfig:
    num_project_candidates: int = 32
    num_worker_candidates: int = 32
    hit_reward: float = 3.0
    miss_penalty: float = -0.5
    score_weight: float = 0.2
    quality_weight: float = 0.25
    winner_bonus: float = 2.0
    finalist_bonus: float = 0.2
    category_match_weight: float = 0.15
    industry_match_weight: float = 0.1
    award_weight: float = 0.01
    urgency_weight: float = 0.02
    project_wait_penalty: float = 0.05
    requester_wait_cost_weight: float = 1.0
    include_truth_in_candidates: bool = False
    mixed_recall: bool = True
    max_steps_per_episode: int | None = None
    release_delay_seconds: int = 1
    # Requester 批量/延迟决策：默认攒够 batch 或临近 deadline 再触发
    requester_immediate_decision: bool = False
    requester_batch_size: int = 8
    requester_deadline_buffer_hours: float = 24.0
    # Worker 候选：纳入 start_date 在未来 lookahead 窗口内的 project（缓解并发 active 过少）
    project_lookahead_hours: float = 168.0
    # utility：最大化可观测利益 proxy；legacy：以历史 hit 为主（对照实验）
    reward_mode: RewardMode = "utility"
    utility_award_weight: float = 0.35
    utility_worker_match_weight: float = 0.25
    utility_worker_skill_weight: float = 0.25
    utility_competition_weight: float = 0.15
    utility_quality_weight: float = 0.40
    utility_expected_score_weight: float = 0.35
    utility_requester_match_weight: float = 0.15
    utility_activity_weight: float = 0.10
    legacy_hit_weight: float = 0.25
    worker_hit_weight: float | None = None
    requester_hit_weight: float | None = None
    requester_finalist_weight: float = 0.0
    requester_wait_pool_penalty: float = 0.0
    requester_repeat_wait_penalty: float = 0.0
    requester_max_waits_per_project: int = 0


@dataclass
class ProjectRuntimeState:
    project: ProjectRecord
    applicants: list[int] = field(default_factory=list)
    applicant_times: dict[int, datetime] = field(default_factory=dict)
    closed: bool = False
    unfilled: bool = False
    winner_id: int | None = None
    closed_at: datetime | None = None
    last_wait_accounted_at: datetime | None = None
    last_requester_decision_at: datetime | None = None
    last_requester_applicant_count: int = 0
    total_wait_cost: float = 0.0
    requester_wait_count: int = 0


@dataclass
class PlatformDecision:
    actor: ActorName
    observation: Observation
    info: dict[str, Any]


@dataclass
class PlatformStep:
    actor: ActorName
    reward: float
    terminated: bool
    decision: PlatformDecision | None
    info: dict[str, Any]


class PlatformSimulationEnv:
    """Serial worker-DQN / requester-DQN environment with shared platform state."""

    def __init__(
        self,
        platform: PlatformDataset,
        config: PlatformEnvConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.platform = platform
        self.dataset = platform.dataset
        self.split = platform.split
        self.config = config or PlatformEnvConfig()
        self.rng = np.random.default_rng(seed)
        ref_time = (
            platform.worker_events[0].timestamp
            if platform.worker_events
            else min(p.start_date for p in platform.projects.values())
        )
        self.encoder = FeatureEncoder(self.dataset, ref_time=ref_time)

        self.project_states: dict[int, ProjectRuntimeState] = {}
        self.worker_busy_project: dict[int, int] = {}
        self._event_heap: list[tuple[datetime, int, PlatformWorkerEvent]] = []
        self._seq = 0
        self.current_time: datetime | None = None
        self.current_decision: PlatformDecision | None = None
        self._current_worker_event: PlatformWorkerEvent | None = None
        self._candidate_project_ids: list[int] = []
        self._current_requester_project_id: int | None = None
        self._requester_candidate_worker_ids: list[int | None] = []
        self._requester_deadline_forced = False
        self.metrics: dict[str, float] = {}

    @property
    def worker_num_actions(self) -> int:
        return self.config.num_project_candidates

    @property
    def requester_num_actions(self) -> int:
        return self.config.num_worker_candidates + 1

    def reset(self, *, seed: int | None = None) -> PlatformDecision:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.project_states = {
            pid: ProjectRuntimeState(project=p, last_wait_accounted_at=p.start_date)
            for pid, p in self.platform.projects.items()
        }
        self.worker_busy_project = {}
        self._event_heap = []
        self._seq = 0
        for ev in self.platform.iter_worker_events():
            self._push_event(ev)
        self.current_time = None
        self.current_decision = None
        self._current_worker_event = None
        self._candidate_project_ids = []
        self._current_requester_project_id = None
        self._requester_candidate_worker_ids = []
        self._requester_deadline_forced = False
        self.metrics = {
            "worker_reward": 0.0,
            "requester_reward": 0.0,
            "platform_reward": 0.0,
            "project_wait_cost": 0.0,
            "project_wait_days": 0.0,
            "worker_hits": 0.0,
            "requester_hits": 0.0,
            "worker_decisions": 0.0,
            "requester_decisions": 0.0,
            "rerouted_workers": 0.0,
            "closed_projects": 0.0,
            "filled_projects": 0.0,
            "unfilled_projects": 0.0,
            "winner_quality_sum": 0.0,
            "winner_count": 0.0,
            "steps": 0.0,
            "worker_recall_opportunities": 0.0,
            "worker_recalls": 0.0,
            "requester_recall_opportunities": 0.0,
            "requester_recalls": 0.0,
            "requester_pool_size_sum": 0.0,
            "requester_waits": 0.0,
            "requester_waited_projects": 0.0,
            "requester_wait_count_sum": 0.0,
            "worker_utility_sum": 0.0,
            "requester_utility_sum": 0.0,
        }
        decision = self._advance()
        if decision is None:
            raise RuntimeError("platform split has no usable decisions")
        return decision

    def step(self, action: int) -> PlatformStep:
        if self.current_decision is None:
            raise RuntimeError("call reset() before step()")
        if self.current_decision.actor == "worker":
            return self._step_worker(action)
        return self._step_requester(action)

    def empty_observation(self, actor: ActorName) -> Observation:
        if actor == "worker":
            return Observation(
                worker_feat=np.zeros(WORKER_FEAT_DIM, dtype=np.float32),
                candidate_feat=np.zeros(
                    (self.config.num_project_candidates, PLATFORM_PROJECT_FEAT_DIM),
                    dtype=np.float32,
                ),
                action_mask=np.zeros(self.config.num_project_candidates, dtype=bool),
            )
        return Observation(
            worker_feat=np.zeros(REQUESTER_CONTEXT_FEAT_DIM, dtype=np.float32),
            candidate_feat=np.zeros(
                (self.config.num_worker_candidates + 1, WORKER_FEAT_DIM),
                dtype=np.float32,
            ),
            action_mask=np.zeros(self.config.num_worker_candidates + 1, dtype=bool),
        )

    def final_metrics(self) -> dict[str, float]:
        steps = max(self.metrics["steps"], 1.0)
        worker_steps = max(self.metrics["worker_decisions"], 1.0)
        requester_steps = max(self.metrics["requester_decisions"], 1.0)
        closed = max(self.metrics["closed_projects"], 1.0)
        winners = max(self.metrics["winner_count"], 1.0)
        return {
            "worker_hit_rate": self.metrics["worker_hits"] / worker_steps,
            "requester_hit_rate": self.metrics["requester_hits"] / requester_steps,
            "worker_reward": self.metrics["worker_reward"],
            "requester_reward": self.metrics["requester_reward"],
            "platform_reward": self.metrics["platform_reward"],
            "project_wait_cost": self.metrics["project_wait_cost"],
            "avg_project_wait_days": self.metrics["project_wait_days"] / closed,
            "filled_project_rate": self.metrics["filled_projects"] / closed,
            "winner_quality": self.metrics["winner_quality_sum"] / winners,
            "rerouted_workers": self.metrics["rerouted_workers"],
            "closed_projects": self.metrics["closed_projects"],
            "unfilled_projects": self.metrics["unfilled_projects"],
            "steps": steps,
            "worker_decisions": self.metrics["worker_decisions"],
            "requester_decisions": self.metrics["requester_decisions"],
            "worker_recall_at_k": (
                self.metrics["worker_recalls"]
                / max(self.metrics["worker_recall_opportunities"], 1.0)
            ),
            "requester_recall_at_k": (
                self.metrics["requester_recalls"]
                / max(self.metrics["requester_recall_opportunities"], 1.0)
            ),
            "platform_reward_per_project": self.metrics["platform_reward"] / closed,
            "platform_reward_per_step": self.metrics["platform_reward"] / steps,
            "avg_requester_pool_size": (
                self.metrics["requester_pool_size_sum"] / requester_steps
            ),
            "requester_waits": self.metrics["requester_waits"],
            "requester_wait_rate": self.metrics["requester_waits"] / requester_steps,
            "requester_waited_project_rate": (
                self.metrics["requester_waited_projects"] / closed
            ),
            "avg_requester_waits_per_project": (
                self.metrics["requester_wait_count_sum"] / closed
            ),
            "avg_worker_utility": self.metrics["worker_utility_sum"] / worker_steps,
            "avg_requester_utility": (
                self.metrics["requester_utility_sum"] / requester_steps
            ),
        }

    def _step_worker(self, action: int) -> PlatformStep:
        assert self._current_worker_event is not None
        ev = self._current_worker_event
        worker_reward = self.config.miss_penalty
        worker_hit = False
        worker_utility = 0.0
        selected_project_id: int | None = None

        if 0 <= action < len(self._candidate_project_ids):
            selected_project_id = self._candidate_project_ids[action]
            state = self.project_states[selected_project_id]
            if not state.closed and ev.worker_id not in state.applicants:
                state.applicants.append(ev.worker_id)
                state.applicant_times[ev.worker_id] = ev.timestamp
                self.worker_busy_project[ev.worker_id] = selected_project_id
                worker_reward, worker_hit, worker_utility = self._worker_reward(
                    ev.worker_id,
                    selected_project_id,
                    ev.timestamp,
                    ev.truth_project_id,
                )

        self.metrics["worker_decisions"] += 1
        self.metrics["worker_reward"] += worker_reward
        self.metrics["platform_reward"] += worker_reward
        self.metrics["worker_hits"] += float(worker_hit)
        self.metrics["worker_utility_sum"] += worker_utility
        self.metrics["steps"] += 1

        info = {
            "actor": "worker",
            "agent_reward": worker_reward,
            "worker_reward": worker_reward,
            "requester_reward": 0.0,
            "platform_reward": worker_reward,
            "project_wait_cost": 0.0,
            "hit": worker_hit,
            "worker_hit": worker_hit,
            "requester_hit": False,
            "worker_id": ev.worker_id,
            "project_id": selected_project_id,
            "timestamp": ev.timestamp.isoformat(),
        }

        if selected_project_id is not None:
            if self._should_trigger_requester(selected_project_id, ev.timestamp):
                self._current_requester_project_id = selected_project_id
                self._requester_deadline_forced = self._is_deadline_forced(
                    selected_project_id
                )
                next_decision = self._make_requester_decision(selected_project_id)
            else:
                next_decision = self._advance()
        else:
            next_decision = self._advance()
        return self._finish_step("worker", worker_reward, info, next_decision)

    def _build_requester_worker_pool(
        self,
        project_id: int,
        applicants: list[int],
        t: datetime,
    ) -> list[int]:
        """申请池 worker 召回：优先保留历史 winner，再按质量/活跃度补齐。"""
        if not applicants:
            return []

        k = self.config.num_worker_candidates
        winner_ids = [
            wid
            for wid in applicants
            if self.platform.outcome_for(project_id, wid).winner
        ]
        ranked = sorted(
            applicants,
            key=lambda wid: (
                -self.dataset.get_worker_quality(wid),
                -self.encoder.worker_history_profile(wid, t).past_count,
                wid,
            ),
        )

        chosen: list[int] = []
        seen: set[int] = set()
        for wid in winner_ids + ranked:
            if wid in seen:
                continue
            seen.add(wid)
            chosen.append(wid)
            if len(chosen) >= k:
                break
        return chosen

    def _step_requester(self, action: int) -> PlatformStep:
        assert self._current_requester_project_id is not None
        pid = self._current_requester_project_id
        state = self.project_states[pid]
        winner_in_applicants = any(
            self.platform.outcome_for(pid, wid).winner for wid in state.applicants
        )
        if winner_in_applicants:
            self.metrics["requester_recall_opportunities"] += 1.0
            candidate_workers = [
                wid
                for wid in self._requester_candidate_worker_ids[1:]
                if wid is not None
            ]
            if any(
                self.platform.outcome_for(pid, wid).winner for wid in candidate_workers
            ):
                self.metrics["requester_recalls"] += 1.0

        requester_reward = self.config.miss_penalty
        requester_hit = False
        requester_utility = 0.0
        selected_worker_id: int | None = None
        wait_cost = 0.0

        wait_allowed = (
            len(self._requester_candidate_worker_ids) > 0
            and self._requester_candidate_worker_ids[0] is None
            and not self._requester_deadline_forced
            and (
                self.config.requester_max_waits_per_project <= 0
                or state.requester_wait_count
                < self.config.requester_max_waits_per_project
            )
        )
        is_wait_action = action == 0 and wait_allowed
        wait_pool_penalty = 0.0
        wait_repeat_penalty = 0.0
        if is_wait_action:
            wait_cost = self._apply_wait_cost(pid, self.current_time_or_project_time(pid))
            wait_pool_penalty = self.config.requester_wait_pool_penalty * float(
                np.log1p(len(state.applicants))
            )
            wait_repeat_penalty = (
                self.config.requester_repeat_wait_penalty * state.requester_wait_count
            )
            requester_reward = 0.0
            if state.requester_wait_count == 0:
                self.metrics["requester_waited_projects"] += 1.0
            state.requester_wait_count += 1
            self.metrics["requester_waits"] += 1.0
        elif 0 <= action < len(self._requester_candidate_worker_ids):
            selected_worker_id = self._requester_candidate_worker_ids[action]
            if selected_worker_id is not None:
                wait_cost = self._apply_wait_cost(
                    pid,
                    self.current_time_or_project_time(pid),
                )
                requester_reward, requester_hit, requester_utility = (
                    self._requester_reward(
                        pid,
                        selected_worker_id,
                        self.current_time_or_project_time(pid),
                    )
                )
                self._close_project(pid, selected_worker_id)
        else:
            wait_cost = self._apply_wait_cost(pid, self.current_time_or_project_time(pid))
            requester_reward = self.config.miss_penalty

        if selected_worker_id is None:
            state.last_requester_decision_at = self.current_time_or_project_time(pid)
            state.last_requester_applicant_count = len(state.applicants)

        requester_wait_penalty = (
            self.config.requester_wait_cost_weight * wait_cost
            + wait_pool_penalty
            + wait_repeat_penalty
        )
        requester_reward -= requester_wait_penalty

        agent_reward = requester_reward
        platform_reward = requester_reward
        self.metrics["requester_decisions"] += 1
        self.metrics["requester_reward"] += requester_reward
        self.metrics["platform_reward"] += platform_reward
        self.metrics["requester_hits"] += float(requester_hit)
        self.metrics["requester_utility_sum"] += requester_utility
        self.metrics["steps"] += 1

        info = {
            "actor": "requester",
            "agent_reward": agent_reward,
            "worker_reward": 0.0,
            "requester_reward": requester_reward,
            "platform_reward": platform_reward,
            "project_wait_cost": wait_cost,
            "requester_wait_penalty": requester_wait_penalty,
            "requester_wait_pool_penalty": wait_pool_penalty,
            "requester_wait_repeat_penalty": wait_repeat_penalty,
            "hit": requester_hit,
            "worker_hit": False,
            "requester_hit": requester_hit,
            "project_id": pid,
            "worker_id": selected_worker_id,
            "timestamp": self.current_time_or_project_time(pid).isoformat(),
            "wait": is_wait_action,
        }

        self._current_requester_project_id = None
        self._requester_candidate_worker_ids = []
        self._requester_deadline_forced = False
        next_decision = self._advance()
        return self._finish_step("requester", agent_reward, info, next_decision)

    def _finish_step(
        self,
        actor: ActorName,
        reward: float,
        info: dict[str, Any],
        next_decision: PlatformDecision | None,
    ) -> PlatformStep:
        self.current_decision = next_decision
        terminated = next_decision is None
        return PlatformStep(
            actor=actor,
            reward=reward,
            terminated=terminated,
            decision=next_decision,
            info=info,
        )

    def _advance(self) -> PlatformDecision | None:
        while True:
            next_worker_time = self._event_heap[0][0] if self._event_heap else None
            due_pid = self._next_due_project(next_worker_time)
            due_time = (
                self.project_states[due_pid].project.deadline if due_pid is not None else None
            )
            buffer_item = self._next_buffer_requester_project(next_worker_time)
            buffer_pid, buffer_time = buffer_item if buffer_item else (None, None)

            candidates: list[tuple[str, datetime, int | None]] = []
            if due_pid is not None and due_time is not None:
                candidates.append(("deadline", due_time, due_pid))
            if buffer_pid is not None and buffer_time is not None:
                candidates.append(("buffer", buffer_time, buffer_pid))
            if next_worker_time is not None:
                candidates.append(("worker", next_worker_time, None))

            if not candidates:
                tail_pid = self._next_due_project(None)
                if tail_pid is None:
                    return None
                continue

            candidates.sort(key=lambda item: (item[1], 0 if item[0] == "deadline" else 1))
            kind, event_time, project_id = candidates[0]

            if kind == "deadline":
                assert project_id is not None
                project = self.project_states[project_id].project
                self.current_time = max(
                    project.deadline,
                    self.current_time or project.deadline,
                )
                if self.project_states[project_id].applicants:
                    self._current_requester_project_id = project_id
                    self._requester_deadline_forced = True
                    return self._make_requester_decision(project_id)
                self._close_unfilled(project_id, self.current_time)
                continue

            if kind == "buffer":
                assert project_id is not None
                self.current_time = max(
                    event_time,
                    self.current_time or event_time,
                )
                self._current_requester_project_id = project_id
                self._requester_deadline_forced = self._is_deadline_forced(project_id)
                return self._make_requester_decision(project_id)

            _, _, ev = heapq.heappop(self._event_heap)
            self.current_time = ev.timestamp
            if ev.worker_id in self.worker_busy_project:
                continue

            decision = self._make_worker_decision(ev)
            if decision is None:
                continue
            return decision

    def _should_trigger_requester(self, project_id: int, t: datetime) -> bool:
        """是否触发 requester 决策（批量/延迟机制）。"""
        if self.config.requester_immediate_decision:
            return True

        state = self.project_states[project_id]
        if not state.applicants:
            return False
        if self._is_deadline_forced(project_id):
            return True
        if len(state.applicants) >= self.config.requester_batch_size:
            return True
        if len(state.applicants) >= self.config.num_worker_candidates:
            return True

        hours_left = max(
            (state.project.deadline - t).total_seconds() / 3600.0,
            0.0,
        )
        return hours_left <= self.config.requester_deadline_buffer_hours

    def _next_buffer_requester_project(
        self,
        before_time: datetime | None,
    ) -> tuple[int, datetime] | None:
        """找最早到达 deadline buffer 且申请池非空、尚未到 deadline 的 project。"""
        if self.config.requester_immediate_decision:
            return None

        best: tuple[int, datetime] | None = None
        buffer_delta = timedelta(hours=self.config.requester_deadline_buffer_hours)
        for pid, state in self.project_states.items():
            if state.closed or not state.applicants:
                continue
            deadline = state.project.deadline
            trigger_at = deadline - buffer_delta
            if trigger_at >= deadline:
                continue
            if (
                state.last_requester_decision_at is not None
                and trigger_at <= state.last_requester_decision_at
                and len(state.applicants) <= state.last_requester_applicant_count
            ):
                continue
            if before_time is not None and trigger_at > before_time:
                continue
            if best is None or trigger_at < best[1] or (
                trigger_at == best[1] and pid < best[0]
            ):
                best = (pid, trigger_at)
        return best

    def _make_worker_decision(
        self,
        ev: PlatformWorkerEvent,
    ) -> PlatformDecision | None:
        candidates, recall_info = self._build_project_candidates(ev)
        if not candidates:
            return None
        if recall_info["truth_active"]:
            self.metrics["worker_recall_opportunities"] += 1.0
        if recall_info["truth_in_candidates"]:
            self.metrics["worker_recalls"] += 1.0
        self._current_worker_event = ev
        self._candidate_project_ids = [p.project_id for p in candidates]
        obs = self._observe_worker(ev, candidates)
        info = {
            "actor": "worker",
            "worker_id": ev.worker_id,
            "timestamp": ev.timestamp.isoformat(),
            "truth_project_id": ev.truth_project_id,
            "candidate_project_ids": list(self._candidate_project_ids),
            "truth_in_candidates": recall_info["truth_in_candidates"],
        }
        decision = PlatformDecision("worker", obs, info)
        self.current_decision = decision
        return decision

    def _make_requester_decision(self, project_id: int) -> PlatformDecision:
        pool_size = len(self.project_states[project_id].applicants)
        self.metrics["requester_pool_size_sum"] += float(pool_size)
        obs = self._observe_requester(project_id)
        info = {
            "actor": "requester",
            "project_id": project_id,
            "timestamp": self.current_time_or_project_time(project_id).isoformat(),
            "deadline_forced": self._requester_deadline_forced,
            "candidate_worker_ids": list(self._requester_candidate_worker_ids),
            "applicant_pool_size": pool_size,
            "batch_triggered": pool_size >= self.config.requester_batch_size,
        }
        decision = PlatformDecision("requester", obs, info)
        self.current_decision = decision
        return decision

    def _build_project_candidates(
        self,
        ev: PlatformWorkerEvent,
    ) -> tuple[list[ProjectRecord], dict[str, bool]]:
        t = ev.timestamp
        k = self.config.num_project_candidates
        active: list[ProjectRecord] = []
        truth_project: ProjectRecord | None = None

        lookahead = timedelta(hours=max(self.config.project_lookahead_hours, 0.0))
        for state in self.project_states.values():
            p = state.project
            if state.closed or t >= p.deadline:
                continue
            if p.start_date > t + lookahead:
                continue
            if ev.worker_id in state.applicants:
                continue
            if (
                self.config.include_truth_in_candidates
                and ev.truth_project_id == p.project_id
            ):
                truth_project = p
                continue
            active.append(p)

        truth_active = ev.truth_project_id is not None and (
            truth_project is not None
            or any(p.project_id == ev.truth_project_id for p in active)
        )
        reserve = 1 if truth_project is not None else 0
        target_size = k - reserve

        if self.config.mixed_recall and not self.config.include_truth_in_candidates:
            chosen = self._mixed_project_recall(ev, active, t, target_size)
        else:
            active.sort(
                key=lambda p: (
                    -self._project_wait_days(p.project_id, t),
                    -p.entry_count,
                    -p.total_awards,
                    p.project_id,
                )
            )
            chosen = active[:target_size]

        if truth_project is not None:
            chosen.append(truth_project)
            self.rng.shuffle(chosen)

        chosen = chosen[:k]
        truth_in_candidates = ev.truth_project_id is not None and any(
            p.project_id == ev.truth_project_id for p in chosen
        )
        return chosen, {
            "truth_active": truth_active,
            "truth_in_candidates": truth_in_candidates,
        }

    def _mixed_project_recall(
        self,
        ev: PlatformWorkerEvent,
        active: list[ProjectRecord],
        t: datetime,
        target_size: int,
    ) -> list[ProjectRecord]:
        """混合召回：匹配 / 热门 / 低等待 / 随机各占一部分槽位。"""
        if not active:
            return []

        profile = self.encoder.worker_history_profile(ev.worker_id, t)
        chosen: list[ProjectRecord] = []
        chosen_ids: set[int] = set()
        match_slots = max(target_size // 2, 1)
        other_slots = max((target_size - match_slots) // 3, 1)

        def take(projects: list[ProjectRecord], limit: int) -> None:
            for project in projects:
                if len(chosen) >= target_size or limit <= 0:
                    return
                if project.project_id in chosen_ids:
                    continue
                chosen_ids.add(project.project_id)
                chosen.append(project)
                limit -= 1

        match_pool = sorted(
            active,
            key=lambda p: (
                -float(
                    profile.dominant_category is not None
                    and p.category == profile.dominant_category
                ),
                -float(
                    profile.dominant_industry_id is not None
                    and p.industry_id == profile.dominant_industry_id
                ),
                -self._project_wait_days(p.project_id, t),
                -p.entry_count,
                p.project_id,
            ),
        )
        take(match_pool, match_slots)

        popularity_pool = sorted(
            active,
            key=lambda p: (-p.entry_count, -p.total_awards, p.project_id),
        )
        take(popularity_pool, other_slots)

        low_wait_pool = sorted(
            active,
            key=lambda p: (
                -self._project_wait_days(p.project_id, t),
                -p.entry_count,
                p.project_id,
            ),
        )
        take(low_wait_pool, other_slots)

        remaining = [p for p in active if p.project_id not in chosen_ids]
        self.rng.shuffle(remaining)
        take(remaining, other_slots)

        if len(chosen) < target_size:
            filler = sorted(
                active,
                key=lambda p: (
                    -self._project_wait_days(p.project_id, t),
                    -p.entry_count,
                    -p.total_awards,
                    p.project_id,
                ),
            )
            take(filler, target_size - len(chosen))
        return chosen

    def _observe_worker(
        self,
        ev: PlatformWorkerEvent,
        candidates: list[ProjectRecord],
    ) -> Observation:
        k = self.config.num_project_candidates
        worker_feat = self.encoder.worker_features(ev.worker_id, ev.timestamp)
        cand_feat = np.zeros((k, PLATFORM_PROJECT_FEAT_DIM), dtype=np.float32)
        mask = np.zeros(k, dtype=bool)
        profile = self.encoder.worker_history_profile(ev.worker_id, ev.timestamp)
        for i, project in enumerate(candidates[:k]):
            cand_feat[i] = self._platform_project_features(
                project,
                ev.worker_id,
                ev.timestamp,
                profile,
            )
            mask[i] = True
        return Observation(worker_feat, cand_feat, mask)

    def _observe_requester(self, project_id: int) -> Observation:
        state = self.project_states[project_id]
        project = state.project
        k = self.config.num_worker_candidates + 1
        t = self.current_time_or_project_time(project_id)
        context_feat = self._platform_project_context_features(
            project,
            t,
            state.applicants,
        )
        cand_feat = np.zeros((k, WORKER_FEAT_DIM), dtype=np.float32)
        mask = np.zeros(k, dtype=bool)

        self._requester_candidate_worker_ids = [None]
        cand_feat[0, -1] = 1.0
        wait_count_allowed = (
            self.config.requester_max_waits_per_project <= 0
            or state.requester_wait_count < self.config.requester_max_waits_per_project
        )
        mask[0] = not self._requester_deadline_forced and wait_count_allowed

        workers = self._build_requester_worker_pool(project_id, state.applicants, t)
        for wid in workers:
            self._requester_candidate_worker_ids.append(wid)

        for i, wid in enumerate(self._requester_candidate_worker_ids[1:], start=1):
            cand_feat[i] = self.encoder.worker_features(wid, t)
            mask[i] = True

        return Observation(context_feat, cand_feat, mask)

    def _platform_project_features(
        self,
        project: ProjectRecord,
        worker_id: int,
        t: datetime,
        profile: Any,
    ) -> np.ndarray:
        dom_cat = profile.dominant_category
        dom_industry = profile.dominant_industry_id
        cat_match = 1.0 if dom_cat is not None and project.category == dom_cat else 0.0
        industry_match = (
            1.0
            if dom_industry is not None and project.industry_id == dom_industry
            else 0.0
        )
        hours_left = max((project.deadline - t).total_seconds() / 3600.0, 0.0)
        hours_open = max((t - project.start_date).total_seconds() / 3600.0, 0.0)
        fill_ratio = self._fill_ratio(project.project_id)
        remaining_ratio = max(1.0 - fill_ratio, 0.0)
        wait_days = self._project_wait_days(project.project_id, t)
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
                fill_ratio,
                remaining_ratio,
                np.log1p(wait_days),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _applicant_pool_quality_stats(
        dataset: Any,
        applicants: list[int],
    ) -> tuple[float, float, float, float]:
        """申请池质量统计：均值、最大值、标准差、top 与均值差距。"""
        if not applicants:
            return 0.0, 0.0, 0.0, 0.0
        qualities = np.array(
            [dataset.get_worker_quality(wid) for wid in applicants],
            dtype=np.float32,
        )
        mean_q = float(qualities.mean())
        max_q = float(qualities.max())
        std_q = float(qualities.std()) if len(qualities) > 1 else 0.0
        return mean_q, max_q, std_q, max_q - mean_q

    def _platform_project_context_features(
        self,
        project: ProjectRecord,
        t: datetime,
        applicants: list[int],
    ) -> np.ndarray:
        hours_left = max((project.deadline - t).total_seconds() / 3600.0, 0.0)
        hours_open = max((t - project.start_date).total_seconds() / 3600.0, 0.0)
        fill_ratio = self._fill_ratio(project.project_id)
        remaining_ratio = max(1.0 - fill_ratio, 0.0)
        wait_days = self._project_wait_days(project.project_id, t)
        pool_mean_q, pool_max_q, pool_std_q, pool_top_gap = (
            self._applicant_pool_quality_stats(self.dataset, applicants)
        )
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
                fill_ratio,
                remaining_ratio,
                np.log1p(wait_days),
                np.log1p(len(applicants)) / 5.0,
                pool_mean_q,
                pool_max_q,
                pool_std_q,
                pool_top_gap,
            ],
            dtype=np.float32,
        )

    def _compute_worker_utility(
        self,
        worker_id: int,
        project_id: int,
        t: datetime,
    ) -> float:
        """参与者利益 proxy：奖金、匹配、类目能力，扣除竞争压力。"""
        project = self.project_states[project_id].project
        profile = self.encoder.worker_history_profile(worker_id, t)
        cat_score, cat_win_rate, _ = self.encoder.worker_category_stats(
            worker_id,
            project.category,
            t,
        )
        cat_match = (
            1.0
            if profile.dominant_category is not None
            and project.category == profile.dominant_category
            else 0.0
        )
        ind_match = (
            1.0
            if profile.dominant_industry_id is not None
            and project.industry_id == profile.dominant_industry_id
            else 0.0
        )
        skill = 0.5 * (cat_score / 5.0) + 0.5 * cat_win_rate
        if profile.past_count == 0:
            skill = 0.5 * (profile.mean_score / 5.0) + 0.5 * profile.win_rate
        competition = min(
            np.log1p(max(project.entry_count, 0.0)) / 10.0,
            1.0,
        )
        return float(
            self.config.utility_award_weight
            * min(np.log1p(max(project.total_awards, 0.0)) / 10.0, 1.0)
            + self.config.utility_worker_match_weight * (0.6 * cat_match + 0.4 * ind_match)
            + self.config.utility_worker_skill_weight * skill
            - self.config.utility_competition_weight * competition
        )

    def _compute_requester_utility(
        self,
        project_id: int,
        worker_id: int,
        t: datetime,
    ) -> float:
        """发布者利益 proxy：质量、预期分数、匹配与活跃度。"""
        project = self.project_states[project_id].project
        profile = self.encoder.worker_history_profile(worker_id, t)
        cat_score, _, _ = self.encoder.worker_category_stats(
            worker_id,
            project.category,
            t,
        )
        expected_score = cat_score if cat_score > 0 else profile.mean_score
        cat_match = (
            1.0
            if profile.dominant_category is not None
            and project.category == profile.dominant_category
            else 0.0
        )
        ind_match = (
            1.0
            if profile.dominant_industry_id is not None
            and project.industry_id == profile.dominant_industry_id
            else 0.0
        )
        activity = min(np.log1p(profile.past_count) / 5.0, 1.0)
        return float(
            self.config.utility_quality_weight
            * self.dataset.get_worker_quality(worker_id)
            + self.config.utility_expected_score_weight * (expected_score / 5.0)
            + self.config.utility_requester_match_weight
            * (0.6 * cat_match + 0.4 * ind_match)
            + self.config.utility_activity_weight * activity
        )

    def _legacy_worker_reward(
        self,
        worker_id: int,
        project_id: int,
        t: datetime,
        truth_project_id: int | None,
        hit: bool,
        outcome: Any,
        profile: Any,
        project: ProjectRecord,
    ) -> float:
        reward = self.config.hit_reward if hit else self.config.miss_penalty
        reward += self.config.score_weight * (outcome.max_revision_score / 5.0)
        if outcome.winner:
            reward += self.config.winner_bonus
        if outcome.finalist:
            reward += self.config.finalist_bonus
        if profile.dominant_category == project.category:
            reward += self.config.category_match_weight
        if profile.dominant_industry_id == project.industry_id:
            reward += self.config.industry_match_weight
        reward += self.config.award_weight * np.log1p(max(project.total_awards, 0.0))
        hours_left = max((project.deadline - t).total_seconds() / 3600.0, 0.0)
        reward += self.config.urgency_weight * min(np.log1p(hours_left) / 10.0, 1.0)
        return float(reward)

    def _legacy_requester_reward(
        self,
        project_id: int,
        worker_id: int,
        outcome: Any,
        profile: Any,
        project: ProjectRecord,
    ) -> float:
        reward = self.config.quality_weight * self.dataset.get_worker_quality(worker_id)
        reward += self.config.score_weight * (outcome.max_revision_score / 5.0)
        if outcome.winner:
            reward += self.config.hit_reward + self.config.winner_bonus
        elif outcome.finalist:
            reward += self.config.finalist_bonus
        if profile.dominant_category == project.category:
            reward += self.config.category_match_weight
        if profile.dominant_industry_id == project.industry_id:
            reward += self.config.industry_match_weight
        return float(reward)

    def _worker_reward(
        self,
        worker_id: int,
        project_id: int,
        t: datetime,
        truth_project_id: int | None,
    ) -> tuple[float, bool, float]:
        project = self.project_states[project_id].project
        outcome = self.platform.outcome_for(project_id, worker_id)
        profile = self.encoder.worker_history_profile(worker_id, t)
        hit = outcome.submitted or truth_project_id == project_id
        utility = self._compute_worker_utility(worker_id, project_id, t)

        if self.config.reward_mode == "utility":
            reward = utility
            if hit:
                reward += (
                    self.config.worker_hit_weight
                    if self.config.worker_hit_weight is not None
                    else self.config.legacy_hit_weight
                )
        else:
            reward = self._legacy_worker_reward(
                worker_id,
                project_id,
                t,
                truth_project_id,
                hit,
                outcome,
                profile,
                project,
            )
        return float(reward), hit, utility

    def _requester_reward(
        self,
        project_id: int,
        worker_id: int,
        t: datetime,
    ) -> tuple[float, bool, float]:
        project = self.project_states[project_id].project
        outcome = self.platform.outcome_for(project_id, worker_id)
        profile = self.encoder.worker_history_profile(worker_id, t)
        hit = outcome.winner
        utility = self._compute_requester_utility(project_id, worker_id, t)

        if self.config.reward_mode == "utility":
            reward = utility
            if hit:
                reward += (
                    self.config.requester_hit_weight
                    if self.config.requester_hit_weight is not None
                    else self.config.legacy_hit_weight
                )
            elif outcome.finalist:
                reward += self.config.requester_finalist_weight
        else:
            reward = self._legacy_requester_reward(
                project_id,
                worker_id,
                outcome,
                profile,
                project,
            )
        return float(reward), hit, utility

    def _apply_wait_cost(self, project_id: int, end_time: datetime) -> float:
        state = self.project_states[project_id]
        start = state.last_wait_accounted_at or state.project.start_date
        if end_time < start:
            return 0.0
        days = (end_time - start).total_seconds() / 86400.0
        cost = self.config.project_wait_penalty * days
        state.last_wait_accounted_at = end_time
        state.total_wait_cost += cost
        self.metrics["project_wait_cost"] += cost
        self.metrics["project_wait_days"] += days
        return float(cost)

    def _close_project(self, project_id: int, winner_id: int) -> None:
        state = self.project_states[project_id]
        if state.closed:
            return
        t = self.current_time_or_project_time(project_id)
        state.closed = True
        state.winner_id = winner_id
        state.closed_at = t
        self.metrics["closed_projects"] += 1
        self.metrics["filled_projects"] += 1
        self.metrics["requester_wait_count_sum"] += state.requester_wait_count
        self.metrics["winner_quality_sum"] += self.dataset.get_worker_quality(winner_id)
        self.metrics["winner_count"] += 1

        for wid in state.applicants:
            if wid == winner_id:
                continue
            self.worker_busy_project.pop(wid, None)
            release_time = t + timedelta(seconds=self.config.release_delay_seconds)
            self._push_event(
                PlatformWorkerEvent(
                    timestamp=release_time,
                    worker_id=wid,
                    truth_project_id=None,
                    source_project_id=project_id,
                    synthetic=True,
                )
            )
            self.metrics["rerouted_workers"] += 1

    def _close_unfilled(self, project_id: int, t: datetime) -> None:
        state = self.project_states[project_id]
        if state.closed:
            return
        self._apply_wait_cost(project_id, t)
        state.closed = True
        state.unfilled = True
        state.closed_at = t
        self.metrics["closed_projects"] += 1
        self.metrics["requester_wait_count_sum"] += state.requester_wait_count
        self.metrics["unfilled_projects"] += 1

    def _next_due_project(self, before_time: datetime | None) -> int | None:
        due: list[tuple[datetime, int]] = []
        for pid, state in self.project_states.items():
            if state.closed:
                continue
            deadline = state.project.deadline
            if before_time is None or deadline <= before_time:
                due.append((deadline, pid))
        if not due:
            return None
        due.sort()
        return due[0][1]

    def _is_deadline_forced(self, project_id: int) -> bool:
        t = self.current_time_or_project_time(project_id)
        return t >= self.project_states[project_id].project.deadline

    def _push_event(self, ev: PlatformWorkerEvent) -> None:
        heapq.heappush(self._event_heap, (ev.timestamp, self._seq, ev))
        self._seq += 1

    def _fill_ratio(self, project_id: int) -> float:
        state = self.project_states[project_id]
        target = max(state.project.entry_count, 1)
        return min(len(state.applicants) / target, 2.0)

    def _project_wait_days(self, project_id: int, t: datetime) -> float:
        project = self.project_states[project_id].project
        start = min(max(t, project.start_date), project.deadline)
        return max((start - project.start_date).total_seconds() / 86400.0, 0.0)

    def current_time_or_project_time(self, project_id: int) -> datetime:
        if self.current_time is not None:
            return self.current_time
        return self.project_states[project_id].project.start_date

    def optimal_worker_action(self, truth_project_id: int | None) -> int | None:
        """BC / 评估标签：utility 模式下为候选内 U_worker 最大；legacy 为 truth index。"""
        if not self._candidate_project_ids:
            return None
        if self.config.reward_mode == "legacy":
            if truth_project_id is None:
                return None
            for idx, pid in enumerate(self._candidate_project_ids):
                if pid == truth_project_id:
                    return idx
            return None

        assert self._current_worker_event is not None
        ev = self._current_worker_event
        best_idx: int | None = None
        best_u = float("-inf")
        for idx, pid in enumerate(self._candidate_project_ids):
            u = self._compute_worker_utility(ev.worker_id, pid, ev.timestamp)
            if u > best_u:
                best_u = u
                best_idx = idx
        return best_idx

    def optimal_requester_action(self, project_id: int) -> int | None:
        """BC 标签：utility 模式下为申请池候选内 U_requester 最大；legacy 为历史 winner。"""
        if self.config.reward_mode == "legacy":
            for idx, wid in enumerate(self._requester_candidate_worker_ids):
                if idx == 0 or wid is None:
                    continue
                if self.platform.outcome_for(project_id, wid).winner:
                    return idx
            return None

        t = self.current_time_or_project_time(project_id)
        best_idx: int | None = None
        best_u = float("-inf")
        for idx, wid in enumerate(self._requester_candidate_worker_ids):
            if idx == 0 or wid is None:
                continue
            u = self._compute_requester_utility(project_id, wid, t)
            if u > best_u:
                best_u = u
                best_idx = idx
        return best_idx


def add_platform_env_cli_args(parser: Any) -> None:
    """向 argparse 注册 platform 环境相关参数。"""
    parser.add_argument(
        "--reward-mode",
        choices=["utility", "legacy"],
        default="utility",
        help="utility=利益 proxy 为主；legacy=历史 hit 为主",
    )
    parser.add_argument("--immediate-requester-decision", action="store_true")
    parser.add_argument("--requester-batch-size", type=int, default=8)
    parser.add_argument("--requester-deadline-buffer-hours", type=float, default=24.0)
    parser.add_argument(
        "--requester-wait-cost-weight",
        type=float,
        default=1.0,
        help="Requester reward 中等待成本的扣减系数；platform_reward 不再额外扣 wait_cost",
    )
    parser.add_argument(
        "--legacy-hit-weight",
        type=float,
        default=0.25,
        help="utility reward 中历史 hit 的额外 bonus；legacy 模式仍使用 hit_reward",
    )
    parser.add_argument(
        "--worker-hit-weight",
        type=float,
        default=None,
        help="utility reward 中 worker 历史投稿 hit 的额外 bonus；默认沿用 legacy-hit-weight",
    )
    parser.add_argument(
        "--requester-hit-weight",
        type=float,
        default=None,
        help="utility reward 中 requester 历史 winner hit 的额外 bonus；默认沿用 legacy-hit-weight",
    )
    parser.add_argument(
        "--requester-finalist-weight",
        type=float,
        default=0.0,
        help="utility reward 中 requester 选中 finalist 的额外 bonus",
    )
    parser.add_argument(
        "--requester-wait-pool-penalty",
        type=float,
        default=0.0,
        help="Requester WAIT 时按 log1p(applicant_pool_size) 额外扣分",
    )
    parser.add_argument(
        "--requester-repeat-wait-penalty",
        type=float,
        default=0.0,
        help="Requester WAIT 时按该 project 已等待次数额外扣分",
    )
    parser.add_argument(
        "--requester-max-waits-per-project",
        type=int,
        default=0,
        help="单个 project 允许 WAIT 的最大次数；0 表示不限制",
    )
    parser.add_argument("--no-mixed-recall", action="store_true")
    parser.add_argument(
        "--project-lookahead-hours",
        type=float,
        default=168.0,
        help="Worker 候选纳入 start_date 在未来 N 小时内的 project；0=仅已开放",
    )
    parser.add_argument(
        "--no-project-lookahead",
        action="store_true",
        help="等价于 --project-lookahead-hours 0",
    )


def platform_env_config_from_args(args: Any, **overrides: Any) -> PlatformEnvConfig:
    """从 CLI 参数构造 PlatformEnvConfig。"""
    cfg = PlatformEnvConfig(
        num_project_candidates=getattr(args, "num_project_candidates", 32),
        num_worker_candidates=getattr(args, "num_worker_candidates", 32),
        include_truth_in_candidates=getattr(args, "include_truth_in_candidates", False),
        mixed_recall=not getattr(args, "no_mixed_recall", False),
        project_wait_penalty=getattr(args, "project_wait_penalty", 0.05),
        reward_mode=getattr(args, "reward_mode", "utility"),
        requester_immediate_decision=getattr(
            args, "immediate_requester_decision", False
        ),
        requester_batch_size=getattr(args, "requester_batch_size", 8),
        requester_deadline_buffer_hours=getattr(
            args, "requester_deadline_buffer_hours", 24.0
        ),
        requester_wait_cost_weight=getattr(args, "requester_wait_cost_weight", 1.0),
        legacy_hit_weight=getattr(args, "legacy_hit_weight", 0.25),
        worker_hit_weight=getattr(args, "worker_hit_weight", None),
        requester_hit_weight=getattr(args, "requester_hit_weight", None),
        requester_finalist_weight=getattr(args, "requester_finalist_weight", 0.0),
        requester_wait_pool_penalty=getattr(
            args, "requester_wait_pool_penalty", 0.0
        ),
        requester_repeat_wait_penalty=getattr(
            args, "requester_repeat_wait_penalty", 0.0
        ),
        requester_max_waits_per_project=getattr(
            args, "requester_max_waits_per_project", 0
        ),
        project_lookahead_hours=(
            0.0
            if getattr(args, "no_project_lookahead", False)
            else getattr(args, "project_lookahead_hours", 168.0)
        ),
        max_steps_per_episode=overrides.pop("max_steps_per_episode", None),
    )
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
