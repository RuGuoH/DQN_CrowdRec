"""
DQN 系列 Q 网络与 Agent。

支持:
- Vanilla DQN (QNetwork)
- Dueling DQN (DuelingQNetwork)
- Double DQN
- 训练日志 / checkpoint（见 TrainingLogger、save_checkpoint）
"""

from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env.worker_env import Observation
from models.training_log import TrainingLogger
from src.features import PROJECT_FEAT_DIM, WORKER_FEAT_DIM

ModelType = Literal["dqn", "dueling"]


def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


class QNetwork(nn.Module):
    """anchor 向量 + K 个候选向量 -> K 维 Q 值。"""

    def __init__(
        self,
        anchor_dim: int = WORKER_FEAT_DIM,
        candidate_dim: int = PROJECT_FEAT_DIM,
        hidden_dim: int = 128,
        num_actions: int = 32,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.anchor_mlp = nn.Sequential(
            nn.Linear(anchor_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.candidate_mlp = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        anchor_feat: torch.Tensor,
        candidate_feat: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, k, _ = candidate_feat.shape
        h_a = self.anchor_mlp(anchor_feat)
        h_c = self.candidate_mlp(candidate_feat.view(b * k, -1)).view(b, k, -1)
        h_a_exp = h_a.unsqueeze(1).expand(-1, k, -1)
        joint = torch.cat([h_a_exp, h_c], dim=-1)
        q = self.head(joint).squeeze(-1)
        if action_mask is not None:
            q = q.masked_fill(~action_mask, -1e9)
        return q

    # 兼容旧接口
    def forward_worker_project(
        self,
        worker_feat: torch.Tensor,
        candidate_feat: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward(worker_feat, candidate_feat, action_mask)


class DuelingQNetwork(nn.Module):
    def __init__(
        self,
        anchor_dim: int = WORKER_FEAT_DIM,
        candidate_dim: int = PROJECT_FEAT_DIM,
        hidden_dim: int = 128,
        num_actions: int = 32,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.anchor_mlp = nn.Sequential(
            nn.Linear(anchor_dim, hidden_dim),
            nn.ReLU(),
        )
        self.candidate_mlp = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        anchor_feat: torch.Tensor,
        candidate_feat: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, k, _ = candidate_feat.shape
        h_a = self.anchor_mlp(anchor_feat).unsqueeze(1).expand(-1, k, -1)
        h_c = self.candidate_mlp(candidate_feat.view(b * k, -1)).view(b, k, -1)
        joint = torch.cat([h_a, h_c], dim=-1)
        v = self.value_head(joint)
        adv = self.adv_head(joint)
        q = (v + adv - adv.mean(dim=1, keepdim=True)).squeeze(-1)
        if action_mask is not None:
            q = q.masked_fill(~action_mask, -1e9)
        return q


def build_q_network(
    model_type: ModelType,
    num_actions: int,
    hidden_dim: int = 128,
    anchor_dim: int = WORKER_FEAT_DIM,
    candidate_dim: int = PROJECT_FEAT_DIM,
) -> nn.Module:
    if model_type == "dueling":
        return DuelingQNetwork(
            anchor_dim=anchor_dim,
            candidate_dim=candidate_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
        )
    return QNetwork(
        anchor_dim=anchor_dim,
        candidate_dim=candidate_dim,
        hidden_dim=hidden_dim,
        num_actions=num_actions,
    )


@dataclass
class Transition:
    obs: dict[str, np.ndarray]
    action: int
    reward: float
    next_obs: dict[str, np.ndarray]
    done: bool


@dataclass
class UpdateResult:
    loss: float
    mean_q: float
    global_step: int


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    target_update_freq: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    hidden_dim: int = 128
    model_type: ModelType = "dqn"
    double_dqn: bool = False
    device: str = "cpu"
    anchor_dim: int = WORKER_FEAT_DIM
    candidate_dim: int = PROJECT_FEAT_DIM


class DQNAgent:
    """DQN / Double DQN / Dueling DQN 统一 Agent。"""

    def __init__(self, num_actions: int, config: DQNConfig | None = None) -> None:
        self.num_actions = num_actions
        self.cfg = config or DQNConfig()
        self.device = torch.device(self.cfg.device)

        self.policy_net = build_q_network(
            self.cfg.model_type,
            num_actions,
            self.cfg.hidden_dim,
            self.cfg.anchor_dim,
            self.cfg.candidate_dim,
        ).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.cfg.lr)
        self.replay = ReplayBuffer(self.cfg.buffer_size)
        self._step_count = 0
        self.last_loss: float | None = None
        self.last_mean_q: float | None = None

    @property
    def epsilon(self) -> float:
        c = self.cfg
        if self._step_count >= c.epsilon_decay_steps:
            return c.epsilon_end
        t = self._step_count / c.epsilon_decay_steps
        return c.epsilon_start + (c.epsilon_end - c.epsilon_start) * t

    @property
    def global_step(self) -> int:
        return self._step_count

    def observe(self, transition: Transition) -> None:
        self.replay.push(transition)

    def select_action(self, obs: Observation, explore: bool = True) -> int:
        valid = np.flatnonzero(obs.action_mask)
        if len(valid) == 0:
            return 0

        if explore and random.random() < self.epsilon:
            return int(random.choice(valid))

        self.policy_net.eval()
        with torch.no_grad():
            w = _to_tensor(obs.worker_feat, self.device).unsqueeze(0)
            c = _to_tensor(obs.candidate_feat, self.device).unsqueeze(0)
            m = torch.as_tensor(obs.action_mask, device=self.device).unsqueeze(0)
            q = self.policy_net(w, c, m)[0].cpu().numpy()
        return int(valid[np.argmax(q[valid])])

    def _batch_obs(
        self, obs_list: list[dict[str, np.ndarray]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w = np.stack([o["worker_feat"] for o in obs_list])
        c = np.stack([o["candidate_feat"] for o in obs_list])
        m = np.stack([o["action_mask"] for o in obs_list])
        return (
            _to_tensor(w, self.device),
            _to_tensor(c, self.device),
            torch.as_tensor(m, device=self.device),
        )

    def update(self) -> float | None:
        result = self.update_with_stats()
        return result.loss if result else None

    def update_with_stats(self) -> UpdateResult | None:
        if len(self.replay) < self.cfg.batch_size:
            return None

        batch = self.replay.sample(self.cfg.batch_size)
        obs_w, obs_c, obs_m = self._batch_obs([t.obs for t in batch])
        next_w, next_c, next_m = self._batch_obs([t.next_obs for t in batch])
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        self.policy_net.train()
        q_all = self.policy_net(obs_w, obs_c, obs_m)
        q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.cfg.double_dqn:
                next_q_policy = self.policy_net(next_w, next_c, next_m)
                next_actions = next_q_policy.argmax(dim=1, keepdim=True)
                next_q_target = self.target_net(next_w, next_c, next_m)
                next_q = next_q_target.gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_w, next_c, next_m).max(dim=1).values
            target = rewards + self.cfg.gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self._step_count += 1
        if self._step_count % self.cfg.target_update_freq == 0:
            self.sync_target()

        loss_val = float(loss.item())
        mean_q = float(q_sa.detach().mean().item())
        self.last_loss = loss_val
        self.last_mean_q = mean_q
        return UpdateResult(loss=loss_val, mean_q=mean_q, global_step=self._step_count)

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def checkpoint_payload(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._step_count,
            "num_actions": self.num_actions,
            "config": asdict(self.cfg),
            "last_loss": self.last_loss,
            "last_mean_q": self.last_mean_q,
        }
        if extra:
            payload["extra"] = extra
        return payload

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(extra), path)

    def load(self, path: str | Path, *, load_optimizer: bool = True) -> dict[str, Any]:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy"])
        self.target_net.load_state_dict(ckpt["target"])
        if load_optimizer and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step_count = int(ckpt.get("step", 0))
        self.last_loss = ckpt.get("last_loss")
        self.last_mean_q = ckpt.get("last_mean_q")
        return ckpt

    def save_checkpoint(
        self,
        logger: TrainingLogger,
        tag: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        path = logger.checkpoint_path(tag)
        self.save(path, extra=extra)
        return path


def save_best_checkpoint(
    agent: DQNAgent,
    logger: TrainingLogger,
    metric: float,
    best_metric: float,
    *,
    higher_is_better: bool = True,
    extra: dict[str, Any] | None = None,
) -> tuple[float, bool]:
    """若 metric 优于历史最佳则保存 best.pt，返回 (best_metric, improved)。"""
    if best_metric == float("-inf") and higher_is_better:
        pass
    improved = (
        metric > best_metric if higher_is_better else metric < best_metric
    )
    if improved:
        agent.save_checkpoint(logger, "best", extra=extra)
        return metric, True
    return best_metric, False
