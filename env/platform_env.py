"""Dynamic two-sided platform simulation for crowdsourcing recommendation."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np

from env.worker_env import Observation
from src.dataset import ProjectRecord
from src.features import PROJECT_FEAT_DIM, WORKER_FEAT_DIM, FeatureEncoder
from src.platform_dataset import PlatformDataset, PlatformWorkerEvent

ActorName = Literal["worker", "requester"]


@dataclass
class PlatformEnvConfig:
    num_project_candidates: int = 32
    num_worker_candidates: int = 32
    hit_reward: float = 1.0
    miss_penalty: float = -0.1
    score_weight: float = 0.2
    quality_weight: float = 0.25
    winner_bonus: float = 0.5
    finalist_bonus: float = 0.2
    category_match_weight: float = 0.15
    industry_match_weight: float = 0.1
    award_weight: float = 0.02
    urgency_weight: float = 0.05
    project_wait_penalty: float = 0.05
    include_truth_in_candidates: bool = False
    max_steps_per_episode: int | None = None
    release_delay_seconds: int = 1


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
    total_wait_cost: float = 0.0


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
                    (self.config.num_project_candidates, PROJECT_FEAT_DIM),
                    dtype=np.float32,
                ),
                action_mask=np.zeros(self.config.num_project_candidates, dtype=bool),
            )
        return Observation(
            worker_feat=np.zeros(PROJECT_FEAT_DIM, dtype=np.float32),
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
        }

    def _step_worker(self, action: int) -> PlatformStep:
        assert self._current_worker_event is not None
        ev = self._current_worker_event
        worker_reward = self.config.miss_penalty
        worker_hit = False
        selected_project_id: int | None = None

        if 0 <= action < len(self._candidate_project_ids):
            selected_project_id = self._candidate_project_ids[action]
            state = self.project_states[selected_project_id]
            if not state.closed and ev.worker_id not in state.applicants:
                state.applicants.append(ev.worker_id)
                state.applicant_times[ev.worker_id] = ev.timestamp
                self.worker_busy_project[ev.worker_id] = selected_project_id
                worker_reward, worker_hit = self._worker_reward(
                    ev.worker_id,
                    selected_project_id,
                    ev.timestamp,
                    ev.truth_project_id,
                )

        self.metrics["worker_decisions"] += 1
        self.metrics["worker_reward"] += worker_reward
        self.metrics["platform_reward"] += worker_reward
        self.metrics["worker_hits"] += float(worker_hit)
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
            self._current_requester_project_id = selected_project_id
            self._requester_deadline_forced = self._is_deadline_forced(
                selected_project_id
            )
            next_decision = self._make_requester_decision(selected_project_id)
        else:
            next_decision = self._advance()
        return self._finish_step("worker", worker_reward, info, next_decision)

    def _step_requester(self, action: int) -> PlatformStep:
        assert self._current_requester_project_id is not None
        pid = self._current_requester_project_id
        state = self.project_states[pid]
        requester_reward = self.config.miss_penalty
        requester_hit = False
        selected_worker_id: int | None = None
        wait_cost = 0.0

        wait_allowed = (
            len(self._requester_candidate_worker_ids) > 0
            and self._requester_candidate_worker_ids[0] is None
            and not self._requester_deadline_forced
        )
        if action == 0 and wait_allowed:
            wait_cost = self._apply_wait_cost(pid, self.current_time_or_project_time(pid))
            requester_reward = 0.0
        elif 0 <= action < len(self._requester_candidate_worker_ids):
            selected_worker_id = self._requester_candidate_worker_ids[action]
            if selected_worker_id is not None:
                wait_cost = self._apply_wait_cost(
                    pid,
                    self.current_time_or_project_time(pid),
                )
                requester_reward, requester_hit = self._requester_reward(
                    pid,
                    selected_worker_id,
                    self.current_time_or_project_time(pid),
                )
                self._close_project(pid, selected_worker_id)
        else:
            wait_cost = self._apply_wait_cost(pid, self.current_time_or_project_time(pid))
            requester_reward = self.config.miss_penalty - wait_cost

        agent_reward = requester_reward - wait_cost
        platform_reward = requester_reward - wait_cost
        self.metrics["requester_decisions"] += 1
        self.metrics["requester_reward"] += requester_reward
        self.metrics["platform_reward"] += platform_reward
        self.metrics["requester_hits"] += float(requester_hit)
        self.metrics["steps"] += 1

        info = {
            "actor": "requester",
            "agent_reward": agent_reward,
            "worker_reward": 0.0,
            "requester_reward": requester_reward,
            "platform_reward": platform_reward,
            "project_wait_cost": wait_cost,
            "hit": requester_hit,
            "worker_hit": False,
            "requester_hit": requester_hit,
            "project_id": pid,
            "worker_id": selected_worker_id,
            "timestamp": self.current_time_or_project_time(pid).isoformat(),
            "wait": action == 0 and wait_allowed,
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
            next_event_time = self._event_heap[0][0] if self._event_heap else None
            due_pid = self._next_due_project(next_event_time)
            if due_pid is not None:
                project = self.project_states[due_pid].project
                self.current_time = max(project.deadline, self.current_time or project.deadline)
                if self.project_states[due_pid].applicants:
                    self._current_requester_project_id = due_pid
                    self._requester_deadline_forced = True
                    return self._make_requester_decision(due_pid)
                self._close_unfilled(due_pid, self.current_time)
                continue

            if not self._event_heap:
                tail_pid = self._next_due_project(None)
                if tail_pid is None:
                    return None
                continue

            _, _, ev = heapq.heappop(self._event_heap)
            self.current_time = ev.timestamp
            if ev.worker_id in self.worker_busy_project:
                continue

            decision = self._make_worker_decision(ev)
            if decision is None:
                continue
            return decision

    def _make_worker_decision(
        self,
        ev: PlatformWorkerEvent,
    ) -> PlatformDecision | None:
        candidates = self._build_project_candidates(ev)
        if not candidates:
            return None
        self._current_worker_event = ev
        self._candidate_project_ids = [p.project_id for p in candidates]
        obs = self._observe_worker(ev, candidates)
        info = {
            "actor": "worker",
            "worker_id": ev.worker_id,
            "timestamp": ev.timestamp.isoformat(),
            "truth_project_id": ev.truth_project_id,
            "candidate_project_ids": list(self._candidate_project_ids),
        }
        decision = PlatformDecision("worker", obs, info)
        self.current_decision = decision
        return decision

    def _make_requester_decision(self, project_id: int) -> PlatformDecision:
        obs = self._observe_requester(project_id)
        info = {
            "actor": "requester",
            "project_id": project_id,
            "timestamp": self.current_time_or_project_time(project_id).isoformat(),
            "deadline_forced": self._requester_deadline_forced,
            "candidate_worker_ids": list(self._requester_candidate_worker_ids),
        }
        decision = PlatformDecision("requester", obs, info)
        self.current_decision = decision
        return decision

    def _build_project_candidates(
        self,
        ev: PlatformWorkerEvent,
    ) -> list[ProjectRecord]:
        t = ev.timestamp
        k = self.config.num_project_candidates
        active: list[ProjectRecord] = []
        truth_project: ProjectRecord | None = None

        for state in self.project_states.values():
            p = state.project
            if state.closed or p.start_date > t or t >= p.deadline:
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

        active.sort(
            key=lambda p: (
                -self._project_wait_days(p.project_id, t),
                -p.entry_count,
                -p.total_awards,
                p.project_id,
            )
        )
        chosen = active[: k - (1 if truth_project is not None else 0)]
        if truth_project is not None:
            chosen.append(truth_project)
            self.rng.shuffle(chosen)
        return chosen[:k]

    def _observe_worker(
        self,
        ev: PlatformWorkerEvent,
        candidates: list[ProjectRecord],
    ) -> Observation:
        k = self.config.num_project_candidates
        worker_feat = self.encoder.worker_features(ev.worker_id, ev.timestamp)
        cand_feat = np.zeros((k, PROJECT_FEAT_DIM), dtype=np.float32)
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
        context_feat = self._platform_project_context_features(project, t)
        cand_feat = np.zeros((k, WORKER_FEAT_DIM), dtype=np.float32)
        mask = np.zeros(k, dtype=bool)

        self._requester_candidate_worker_ids = [None]
        cand_feat[0, -1] = 1.0
        mask[0] = not self._requester_deadline_forced

        workers = sorted(
            state.applicants,
            key=lambda wid: (
                -self.dataset.get_worker_quality(wid),
                -self.encoder.worker_history_profile(wid, t).past_count,
                wid,
            ),
        )
        for wid in workers[: self.config.num_worker_candidates]:
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
        cat_match = 1.0 if dom_cat is not None and project.category == dom_cat else 0.0
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
                fill_ratio,
                remaining_ratio,
                np.log1p(wait_days),
            ],
            dtype=np.float32,
        )

    def _platform_project_context_features(
        self,
        project: ProjectRecord,
        t: datetime,
    ) -> np.ndarray:
        hours_left = max((project.deadline - t).total_seconds() / 3600.0, 0.0)
        hours_open = max((t - project.start_date).total_seconds() / 3600.0, 0.0)
        fill_ratio = self._fill_ratio(project.project_id)
        remaining_ratio = max(1.0 - fill_ratio, 0.0)
        wait_days = self._project_wait_days(project.project_id, t)
        applicants = len(self.project_states[project.project_id].applicants)
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
                np.log1p(applicants) / 5.0,
            ],
            dtype=np.float32,
        )

    def _worker_reward(
        self,
        worker_id: int,
        project_id: int,
        t: datetime,
        truth_project_id: int | None,
    ) -> tuple[float, bool]:
        project = self.project_states[project_id].project
        outcome = self.platform.outcome_for(project_id, worker_id)
        profile = self.encoder.worker_history_profile(worker_id, t)
        hit = outcome.submitted or truth_project_id == project_id
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
        return float(reward), hit

    def _requester_reward(
        self,
        project_id: int,
        worker_id: int,
        t: datetime,
    ) -> tuple[float, bool]:
        project = self.project_states[project_id].project
        outcome = self.platform.outcome_for(project_id, worker_id)
        profile = self.encoder.worker_history_profile(worker_id, t)
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
        return float(reward), outcome.winner

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
