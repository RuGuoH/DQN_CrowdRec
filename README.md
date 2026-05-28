# 众包任务推荐 · 动态双边 DQN 实验

> 使用 AI 协作请先阅读 [agent.md](agent.md)。实验报告大纲见 [docs/report_outline.md](docs/report_outline.md)，模型过程记录见 [docs/experiment_process.md](docs/experiment_process.md)。

## 当前主线

本仓库现在以 **动态双边平台仿真** 为主实验：

- worker 到达时，Worker-DQN 从当前开放且未关闭的 project 中推荐 1 个任务。
- project 收到候选 worker 后，Requester-DQN 决定 `WAIT` 继续等待，或从申请池中选出 winner。
- project 选出 winner 后关闭；未中标 worker 会被释放并重新进入推荐队列。
- reward 同时记录 worker 收益、requester 收益、platform 总收益和 project 等候时间成本。

旧的独立 worker/requester 实验仍保留为 legacy 对照入口。

## 目录结构

```text
├── src/
│   ├── dataset.py              # 原始 Crowdspring 数据读取、时间划分、缓存
│   ├── platform_dataset.py     # 动态平台统一事件和 outcome 索引
│   └── features.py             # worker/project 特征
├── env/
│   ├── platform_env.py         # 动态双边联合仿真环境（主线）
│   ├── worker_env.py           # legacy: 独立参与者侧环境
│   └── requester_env.py        # legacy: 独立请求者侧环境
├── models/
│   ├── dqn.py                  # Vanilla / Dueling / Double DQN
│   ├── platform_training.py    # 双 agent 异步训练/评估循环
│   ├── platform_baselines.py   # 动态平台启发式基线
│   └── training_log.py
├── scripts/
│   ├── train_platform_dqn.py
│   ├── evaluate_platform.py
│   ├── run_platform_baselines.py
│   └── train_worker_dqn.py / train_requester_dqn.py / evaluate.py / run_baselines.py  # legacy
└── runs/
```

## 快速开始

```bash
pip install -r requirements.txt

# 数据检查
python -m src.dataset --max-projects 50

# 动态平台启发式基线 smoke
python scripts/run_platform_baselines.py --split train --max-projects 50 --max-steps 100

# 动态平台双 DQN 短训练
python scripts/train_platform_dqn.py --max-projects 50 --episodes 1 --max-steps 100

# 动态平台 DQN test 评估
python scripts/evaluate_platform.py --split test --max-projects 0 \
  --worker-policy dqn \
  --requester-policy dqn \
  --worker-checkpoint runs/platform/.../checkpoints/worker_best.pt \
  --requester-checkpoint runs/platform/.../checkpoints/requester_best.pt
```

默认主实验不强制把真实标签注入候选集。如需诊断候选集排序上限，可加 `--include-truth-in-candidates`。

## 动态平台 MDP

**Worker-DQN**

| 要素 | 定义 |
|------|------|
| 状态 | 当前 worker 历史画像 + K 个 active project 动态特征 + mask |
| 动作 | 从候选 project 中推荐 1 个 |
| reward | 历史命中、类目/行业匹配、奖金、剩余时间、score、winner/finalist |

**Requester-DQN**

| 要素 | 定义 |
|------|------|
| 状态 | 当前 project 动态状态 + 申请池 worker 特征 + mask |
| 动作 | `WAIT` 或选择 1 个 worker 为 winner |
| reward | worker quality、score、winner/finalist、匹配度，扣除 project 等候成本 |

输出指标包括：

```text
worker_hit_rate, requester_hit_rate, worker_reward, requester_reward,
platform_reward, project_wait_cost, avg_project_wait_days,
filled_project_rate, winner_quality, rerouted_workers,
closed_projects, unfilled_projects, steps
```

## Legacy 对照

以下入口保留用于复现旧实验，但不再作为报告主线：

```bash
python scripts/train_worker_dqn.py --max-projects 50 --episodes 10
python scripts/train_requester_dqn.py --max-projects 50 --episodes 10
python scripts/run_baselines.py --side worker --split test --max-projects 50
python scripts/run_baselines.py --side requester --split test --max-projects 50
```
