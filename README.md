# 众包任务推荐 · 强化学习大作业

> **使用 AI 协作**：请先阅读 [agent.md](agent.md)（项目目标、实现现状、Agent 行为标准与提示词模板）。

## 目录结构

```
├── configs/default.yaml   # 配置
├── data/data/             # 原始数据（已有）
├── src/
│   ├── config.py
│   └── dataset.py         # 数据加载与事件流
├── env/                   # MDP 环境
├── models/                # DQN
├── scripts/               # 训练脚本
├── cache/                 # 数据缓存（自动生成）
└── requirements.txt
```

## 快速开始

```bash
pip install -r requirements.txt

# 快速检查（默认加载 50 个项目）
python -m src.dataset

# 加载全量（较慢，首次会写 cache）
python -m src.dataset --max-projects 0
```

## 数据 API 示例

```python
from src.dataset import build_dataset

ds = build_dataset()
for ev in ds.iter_worker_events("train"):
    active = ds.active_projects_at(ev.timestamp)
    ...
```

## 环境与 DQN

```bash
# 快速验证
python scripts/smoke_env.py
python scripts/smoke_requester.py

# 参与者侧训练（日志与 checkpoint 写入 runs/worker/）
python scripts/train_worker_dqn.py --max-projects 50 --episodes 10

# 请求者侧训练
python scripts/train_requester_dqn.py --max-projects 50 --episodes 10

# Dueling + Double DQN
python scripts/train_worker_dqn.py --model dueling --double-dqn

# 评估与基线对比（test 集）
python scripts/evaluate.py --side worker --split test --policy random --max-projects 50
python scripts/evaluate.py --side worker --split test --policy dqn \
  --checkpoint runs/worker/.../checkpoints/best.pt

# 批量跑所有基线（输出 comparison.csv）
python scripts/run_baselines.py --side worker --split test --max-projects 50
python scripts/run_baselines.py --side requester --split test --max-projects 50
```

实验报告大纲见 [docs/report_outline.md](docs/report_outline.md)。

训练输出目录示例：`runs/worker/worker_dqn_20260518_120000/`
- `metrics.csv`：每 episode 的 reward / hit_rate / loss
- `config.json`：超参快照
- `checkpoints/best.pt`、`ep0005.pt`、`final.pt`

### MDP 概要

**参与者**（`env/worker_env.py`）

| 要素 | 定义 |
|------|------|
| 状态 | `worker_feat`(8) + `candidate_feat`(K×10) + `action_mask` |
| 动作 | 从 K 个候选项目中选一个 |
| 奖励 | 命中真实投稿项目 + 质量/获奖加成 |

**请求者**（`env/requester_env.py`）

| 要素 | 定义 |
|------|------|
| 状态 | `worker_feat` 槽位存项目上下文(10) + `candidate_feat`(K×8) 为 worker |
| 动作 | 从 K 个候选 worker 中推荐 1 人 |
| 奖励 | 命中真实投稿 worker + 分数/质量/获奖加成 |
