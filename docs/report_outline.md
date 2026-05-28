# 强化学习大作业实验报告大纲

> 题目：基于 DQN 的动态双边众包任务推荐
> 主实验：Worker-DQN 任务推荐 + Requester-DQN 动态选人/等待

## 摘要

- 研究问题：动态变化的 worker 和 project 如何进行双边推荐。
- 方法：离线历史日志、统一平台事件流、动态联合仿真、两个串行 DQN。
- 指标：worker hit、requester hit、platform reward、project 等候成本、filled project rate。

## 1 引言

- 众包平台中 worker 到达和 project 发布/截止都随时间变化。
- 平台需要同时兼顾参与者收益和发布者获得高质量稿件的收益。
- 本文把旧的双端独立候选排序改为一个动态平台仿真：worker 被推荐项目，project 再决定继续等待或选出 winner，未中标 worker 回流。

## 2 数据与统一事件格式

- 原始数据：`project_list.csv`、`worker_quality.csv`、`project/`、`entry/`。
- 划分：按 project `start_date` 排序后 70% / 15% / 15%。
- 新增 `PlatformDataset`：
  - worker 到达事件：按 `entry_created_at` 排序。
  - project 状态表：start、deadline、entry_count、奖金、类目、行业。
  - worker-project outcome：是否历史投稿、score、winner、finalist。
  - synthetic release event：project 关闭后，未中标 worker 重新进入队列。

## 3 方法

### 3.1 动态平台流程

```text
worker 到达
  -> Worker-DQN 推荐 1 个 active project
  -> worker 进入 project 申请池并暂时不可再推荐
  -> Requester-DQN 选择 WAIT 或选 winner
  -> project 选出 winner 后关闭
  -> 未中标 worker 释放并重新推荐新任务
```

### 3.2 Worker-DQN

| 要素 | 定义 |
|------|------|
| 状态 | worker 历史画像 + K 个 active project 动态特征 + mask |
| 动作 | 推荐 1 个 project |
| reward | 历史命中、类目/行业匹配、奖金、剩余时间、score、winner/finalist |

### 3.3 Requester-DQN

| 要素 | 定义 |
|------|------|
| 状态 | project 动态状态 + 申请池 worker 特征 + mask |
| 动作 | `WAIT` 或选择 1 个 worker 为 winner |
| reward | worker quality、score、winner/finalist、匹配度，扣除 project 等候成本 |

### 3.4 Q 网络

- 两个 DQN 共用 `anchor MLP + candidate MLP -> per-action Q` 结构。
- Worker-DQN：anchor=worker，candidate=project。
- Requester-DQN：anchor=project，candidate=worker，动作 0 为 `WAIT`。
- 支持 Vanilla DQN、Double DQN、Dueling DQN。

## 4 实验设置

### 4.1 复现命令

```bash
python -m src.dataset --max-projects 0

python scripts/train_platform_dqn.py --max-projects 0 --episodes 10 --device cuda

python scripts/run_platform_baselines.py --split test --max-projects 0

python scripts/evaluate_platform.py --split test --max-projects 0 \
  --worker-policy dqn --requester-policy dqn \
  --worker-checkpoint runs/platform/.../checkpoints/worker_best.pt \
  --requester-checkpoint runs/platform/.../checkpoints/requester_best.pt
```

### 4.2 指标

| 指标 | 含义 |
|------|------|
| `worker_hit_rate` | Worker-DQN 推荐 project 是否对应历史投稿 outcome |
| `requester_hit_rate` | Requester-DQN 选择 worker 是否为历史 winner |
| `worker_reward` | 参与者侧累计收益 |
| `requester_reward` | 发布者侧累计收益，不含等待成本 |
| `platform_reward` | 双边收益扣除 project 等待成本 |
| `project_wait_cost` | project 从发布到 winner/关闭的等待成本 |
| `filled_project_rate` | 成功选出 winner 的 project 比例 |
| `winner_quality` | 被选中 winner 的平均质量 |
| `rerouted_workers` | 未中标后被重新推荐的 worker 数 |

### 4.3 基线

| 组合 | Worker 策略 | Requester 策略 |
|------|-------------|----------------|
| Random-Wait | random_project | wait_until_deadline |
| Popularity-Quality | popularity | worker_quality |
| Category-Category | category_match | worker_category_match |
| Industry-Industry | industry_match | worker_industry_match |
| Award-Quality | award | worker_quality |
| LowWait-Quality | low_wait_project | worker_quality |
| JointHeuristic | category_match + low_wait_project | worker_quality |

### 4.4 结果表

数据来源：`runs/platform_baselines/platform_test_no_truth/comparison.csv` 和 `runs/platform/.../metrics.csv`。

| 方法 | Worker Hit | Requester Hit | Platform Reward | Wait Cost | Filled Rate | Winner Quality | Rerouted |
|------|------------|---------------|-----------------|-----------|-------------|----------------|----------|
| Random-Wait | | | | | | | |
| Popularity-Quality | | | | | | | |
| Category-Category | | | | | | | |
| JointHeuristic | | | | | | | |
| Platform DQN | | | | | | | |

## 5 讨论

- 动态性：候选项目随 start/deadline/closed 状态变化，worker 会因申请池和未中标回流而动态变化。
- 双边目标：Worker-DQN 与 Requester-DQN 保持各自目标，但通过同一平台状态和 platform reward 互相影响。
- 离线局限：未观察到的 worker-project 组合只能用 proxy reward，winner/finalist 信号稀疏。
- legacy 对照：旧 worker/requester 独立实验可用于说明为什么候选集排序不足以回应动态平台要求。

## 6 结论

- 总结动态平台 DQN 相比启发式基线的表现。
- 说明 project 等待成本和 worker 回流机制对结果的影响。
- 展望：更真实的容量约束、多目标 RL、离线策略评估。
