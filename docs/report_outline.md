# 强化学习大作业实验报告大纲

> 题目：基于 DQN 的动态双边众包任务推荐
> 主实验：Worker-DQN 任务推荐 + Requester-DQN 动态选人/等待
> 代码主线：`PlatformDataset` + `PlatformSimulationEnv`

## 摘要

- 研究问题：众包平台中 worker 到达、project 发布和 project 截止都随时间动态变化，平台需要同时为参与者推荐合适任务，并帮助请求者获得高质量投稿。
- 方法：基于 Crowdspring 历史日志构建离线动态双边仿真环境，将 worker 到达、project 申请池、requester 等待/选人、project 关闭和未中标 worker 回流统一到同一平台状态中；分别训练 Worker-DQN 与 Requester-DQN。
- 主要指标：`worker_hit_rate`、`requester_hit_rate`、`worker_reward`、`requester_reward`、`platform_reward`、`project_wait_cost`、`filled_project_rate`、`winner_quality`、`rerouted_workers`。
- 主要结论：全量完整 episode 训练后，Platform DQN 在 test 集取得 `platform_reward=265.1064`、`worker_hit_rate=0.0593`、`requester_hit_rate=0.0070`，高于多数正常启发式策略的 platform reward；但 `random_project + wait_until_deadline` 因大量等待和 worker 回流得到异常高累计 reward，需要结合 `rerouted_workers=4517`、`unfilled_projects=30` 和高等待成本单独解释。

## 1 引言

### 1.1 背景

- 众包平台连接两类动态主体：参与者 worker 和发布者 requester/project。
- worker 并非一次性静态集合，而是随投稿时间到达；project 也有发布时间、截止时间、申请池和关闭状态。
- 如果只做静态候选集排序，无法表达“项目被选择后进入申请池”“发布者继续等待或选出 winner”“未中标 worker 重新推荐”等动态过程。

### 1.2 问题定义

- 每次 worker 到达时，平台只推荐 1 个 project。
- 每个 project 收到申请 worker 后，requester 可以选择 `WAIT` 继续等待，或从申请池中选出 1 个 winner。
- winner 产生后 project 关闭；winner 不回流，其他申请池 worker 被释放并重新进入推荐队列。
- 优化目标包括：
  - 参与者侧：推荐 worker 更可能感兴趣、收益更高、匹配度更高的 project。
  - 请求者侧：帮助 project 更及时获得高质量 worker。
  - 平台侧：在双边收益基础上扣除 project 等候时间成本。

### 1.3 本文工作

- 构造统一平台事件格式 `PlatformDataset`。
- 构造动态双边平台仿真环境 `PlatformSimulationEnv`。
- 建立 Worker-DQN 与 Requester-DQN 两个串行智能体。
- 设计基于历史 outcome 的离线 reward，并加入 project 等候时间成本。
- 与随机、流行度、类目匹配、行业匹配、奖金优先、低等待项目、worker quality 等启发式基线比较。

## 2 数据分析与事件构造

### 2.1 原始数据

| 数据源 | 主要字段 | 用途 |
|--------|----------|------|
| `project_list.csv` | `project_id`, `entry_count` | project 总览和投稿量 |
| `project/project_*.txt` | `start_date`, `deadline`, `category`, `sub_category`, `industry`, `total_awards`, `average_score`, `featured` | project 状态与候选 project 特征 |
| `entry/entry_*.txt` | `author`, `entry_created_at`, `max_revision_score`, `winner`, `finalist`, `withdrawn` | worker 到达事件和 worker-project outcome |
| `worker_quality.csv` | `worker_id`, `quality` | worker 质量特征和 requester reward |

### 2.2 时间划分

- 按 project `start_date` 排序后划分 train / val / test，比例为 70% / 15% / 15%。
- 当前全量数据规模：`projects=2447`，`entries=186605`，非撤回投稿 `116274`，`workers_with_entries=1753`，`workers_with_quality=1653`，`industries=37`，`train_projects=1712`，`val_projects=367`，`test_projects=368`。
- 数据时间范围：project `start_date` 从 `2018-01-01` 到 `2019-02-28`，deadline 从 `2018-01-03` 到 `2019-03-03`。
- 描述性统计：project 平均投稿数 `76.29`，中位数 `59`，最大值 `661`；平均奖金 `285.53`，中位数 `200`，最大值 `2900`；平均持续时间 `9.30` 天，中位数 `7` 天；worker quality 平均 `0.7992`，中位数 `0.78`。
- 数据分析输出：`runs/report_full_20260529/data_analysis/summary.json`；图表包括 `project_start_month.png`、`entry_count_distribution.png`、`award_distribution.png`、`worker_quality_distribution.png`、`project_duration_distribution.png`。
- 采用时间划分的原因：避免用未来 project 或未来投稿行为训练模型后再评估过去样本。

### 2.3 平台事件格式

统一事件层由 `src/platform_dataset.py` 构造：

- worker 到达事件：
  - `timestamp = entry_created_at`
  - `worker_id = author`
  - `truth_project_id = 历史真实投稿 project`
  - 按 `(timestamp, worker_id, truth_project_id)` 排序。
- project 状态：
  - `start_date`、`deadline`、`closed`、`unfilled`、`applicants`、`winner_id`、`last_wait_accounted_at`。
- worker-project outcome：
  - `submitted`、`entry_time`、`max_revision_score`、`winner`、`finalist`。
- synthetic release event：
  - project 关闭后，非 winner worker 在 `closed_at + release_delay` 时重新进入 worker 到达队列。

### 2.4 特征构造思路

特征构造原则：

- 只使用当前时刻 `t` 之前可观测的历史，避免未来信息泄漏。
- 同时保留静态属性和动态状态。
- 将连续长尾变量使用 `log1p` 缩放。
- 使用 `action_mask` 屏蔽非法候选，保证 DQN 不选择已关闭 project、deadline 后 project 或不存在的候选槽位。

Worker 特征 `x_w(t) in R^12`：

| 特征 | 含义 |
|------|------|
| `quality` | worker 质量分 |
| `log1p(past_count)` | 历史投稿活跃度 |
| `mean_score / 5` | 历史平均投稿分数 |
| `win_rate` | 历史 winner 比例 |
| `finalist_rate` | 历史 finalist 比例 |
| `dominant_category` | 历史主导类目 |
| `dominant_industry` | 历史主导行业 |
| `dominant_category_share` | 主导类目占比，衡量兴趣集中度 |
| `dominant_industry_share` | 主导行业占比 |
| `log1p(gap_hours)` | 距上次投稿时间 |
| `log1p(recent_30d_count)` | 近 30 天活跃度 |
| `log1p(past_count) / 10` | 归一化历史规模 |

Project 特征 `x_p(t,w) in R^13`：

| 特征 | 含义 |
|------|------|
| `category`, `sub_category`, `industry` | 任务主题和行业 |
| `log1p(entry_count)` | 历史总投稿需求/热度 |
| `log1p(total_awards)` | 奖金规模 |
| `average_score / 5` | 项目历史平均质量 |
| `featured` | 是否精选 |
| `log1p(hours_left)` | 距 deadline 剩余时间 |
| `log1p(hours_open)` | 已开放时间 |
| `category_match` | project 类目是否匹配 worker 历史主类目 |
| `fill_ratio` | 当前申请池人数 / 目标投稿数 |
| `remaining_ratio` | 剩余需求比例 |
| `log1p(wait_days)` | project 已等待天数 |

Requester 侧 project context 同样为 13 维，但最后一维使用 `log1p(applicants)` 表示当前申请池规模。

## 3 数学建模

### 3.1 动态平台 MDP

将平台看作一个事件驱动的马尔可夫决策过程：

`M = (S, A, P, R, gamma)`

其中全局状态 `S_t` 包括：

- 当前时间 `t`。
- 所有 project 的运行状态：是否发布、是否截止、是否关闭、申请池、winner、等待时间。
- worker 状态：空闲、忙于某个 project、是否已被释放。
- 事件队列：真实 worker 到达事件和 synthetic release event。
- 历史 outcome 映射：用于离线 reward 计算。

实际 DQN 不直接输入完整 `S_t`，而是输入局部观测：

- Worker-DQN 观测 `o^W_t = (x_w(t), X^P_t, m^W_t)`。
- Requester-DQN 观测 `o^R_t = (x_p(t), X^W_t, m^R_t)`。

其中 `X^P_t` 是候选 project 特征矩阵，`X^W_t` 是候选 worker 特征矩阵，`m` 是动作 mask。

### 3.2 时间推进

平台按如下时间顺序推进：

1. 将所有历史投稿转为 worker 到达事件，按 `entry_created_at` 排序。
2. 每次取事件队列中最早 worker 到达时间 `t_next`。
3. 在处理该 worker 前，检查是否存在 `deadline <= t_next` 且未关闭的 project。
4. 若存在到期 project：
   - 若申请池非空，触发 Requester-DQN 的 deadline-forced 决策，此时 `WAIT` 非法。
   - 若申请池为空，project 关闭为 `unfilled`。
5. 若无到期 project，则处理 worker 到达并触发 Worker-DQN。
6. 若 requester 选出 winner，则关闭 project，并为非 winner worker 生成 synthetic release event。

这使任务创建时间、到期时间、worker 到达时间处在同一个时间轴中。

### 3.3 Worker-DQN 建模

候选 project 集合：

`C^W_t(w) = {p | start_p <= t < deadline_p, p 未关闭, w 未在 p 的申请池中}`

状态：

`o^W_t = (x_w(t), {x_p(t,w)}_{p in C^W_t}, m^W_t)`

动作：

`a^W_t in {1, ..., K}`，表示从候选集中选择 1 个 project 推荐给当前 worker。

状态转移：

- 设选择 project 为 `p = C^W_t[a]`。
- 将 `w` 加入 `A_p` 申请池。
- 标记 `busy(w) = p`。
- 立即触发该 project 的 requester 决策，或继续推进到下一个时间事件。

Worker reward：

```text
r^W_t =
  hit_reward * I(submitted(w,p))
  + miss_penalty * I(not submitted(w,p))
  + lambda_score * score(w,p) / 5
  + lambda_win * I(winner(w,p))
  + lambda_fin * I(finalist(w,p))
  + lambda_cat * I(dom_cat(w,t) = cat(p))
  + lambda_ind * I(dom_ind(w,t) = ind(p))
  + lambda_award * log(1 + award(p))
  + lambda_urgency * urgency(p,t)
```

其中 `submitted/winner/finalist/score` 来自历史 outcome；类目和行业匹配来自 worker 在 `t` 之前的历史画像。

### 3.4 Requester-DQN 建模

申请池：

`A_p(t) = {w | w 已被推荐到 p 且 p 未关闭}`

候选 worker 集合：

`C^R_t(p) = {WAIT} union topK(A_p(t))`

其中 `topK` 当前按 worker quality、历史活跃度排序召回。

状态：

`o^R_t = (x_p^{ctx}(t), {x_w(t)}_{w in C^R_t}, m^R_t)`

动作：

- `a^R_t = 0`：`WAIT`，deadline 前合法。
- `a^R_t = i`：选择第 `i` 个 worker 为 winner。

状态转移：

- 若选择 `WAIT`：project 保持开放，累计等待成本，平台继续处理后续 worker 到达或 deadline。
- 若选择 worker `w`：project 关闭，`winner_id = w`；winner 不回流，其他申请池 worker 释放并重新进入事件队列。
- 若到 deadline 且申请池为空：project 关闭为 `unfilled`。

Requester reward：

```text
r^R_t =
  lambda_q * quality(w)
  + lambda_score * score(w,p) / 5
  + hit_reward * I(winner(w,p))
  + winner_bonus * I(winner(w,p))
  + finalist_bonus * I(finalist(w,p))
  + lambda_cat * I(dom_cat(w,t) = cat(p))
  + lambda_ind * I(dom_ind(w,t) = ind(p))
```

当前 reward 设计中，普通非 winner 不再额外受到 `miss_penalty=-0.1`；这避免把“质量较好但不是历史 winner”的选择过度惩罚。

等待成本：

```text
cost_wait(p, t0, t1) = beta_wait * (t1 - t0) / 1 day
```

Requester agent 实际学习信号：

`r^{R-agent}_t = r^R_t - cost_wait`

平台总 reward：

`r^{platform}_t = r^W_t + r^R_t - cost_wait`

### 3.5 终止条件

episode 在以下情况终止：

- worker 事件队列为空，且所有 project 都已关闭或无法继续推进。
- 达到 `max_steps_per_episode`。
- 数据 split 内所有可处理事件完成。

## 4 Q 函数与 DQN 算法

### 4.1 Q 函数定义

对每个 agent，目标是学习动作价值函数：

`Q_theta(o_t, a_t) = E[sum_{k=0}^{infty} gamma^k r_{t+k} | o_t, a_t]`

由于每一步的候选集合不同，不能使用固定语义的动作编号表示所有 project 或 worker。因此采用候选集打分式 Q 网络：

```text
h_anchor = MLP_anchor(anchor_feat)
h_i      = MLP_candidate(candidate_feat_i)
Q(o, a_i) = MLP_head([h_anchor, h_i])
```

Worker-DQN：

```text
anchor_feat    = worker_feat in R^12
candidate_feat = project_feat_i in R^13
Q_W(o^W, a_i)  = Q(project_i | worker, platform state)
```

Requester-DQN：

```text
anchor_feat    = project_context_feat in R^13
candidate_feat = worker_feat_i in R^12
Q_R(o^R, a_i)  = Q(select worker_i or WAIT | project, applicants)
```

非法候选动作使用 mask：

`Q(o,a) = -1e9, if m(a)=0`

### 4.2 DQN 更新

经验回放中存储：

`(o_t, a_t, r_t, o_{t+1}, done)`

普通 DQN 的 TD target：

`y_t = r_t + gamma * max_{a'} Q_{theta^-}(o_{t+1}, a') * (1 - done)`

损失函数：

`L(theta) = Huber(Q_theta(o_t, a_t) - y_t)`

其中 `theta^-` 是 target network 参数，周期性从 policy network 同步。

### 4.3 Double DQN

Double DQN 将动作选择和动作估值拆开：

```text
a* = argmax_{a'} Q_theta(o_{t+1}, a')
y_t = r_t + gamma * Q_{theta^-}(o_{t+1}, a*) * (1 - done)
```

这样可以减少普通 DQN 中 `max` 操作带来的 Q 值过估计。

### 4.4 Dueling DQN

Dueling 网络将价值分解为状态价值和动作优势：

`Q(o,a) = V(o,a-context) + A(o,a) - mean_{a'} A(o,a')`

在当前实现中，`V` 与 `A` 都基于 anchor-candidate 拼接后的表示计算，用于提高候选动作价值估计的稳定性。

### 4.5 双 agent 异步训练

Worker-DQN 和 Requester-DQN 各自有：

- 独立 Q 网络。
- 独立 target network。
- 独立 replay buffer。
- 独立 epsilon-greedy 探索。
- 独立 checkpoint。

由于两个 agent 在同一平台环境中交替决策，一个 worker 决策后往往紧接 requester 决策。训练时不是把 requester 状态作为 worker 的 next state，而是：

- worker transition 连接到下一次 worker 决策状态。
- requester transition 连接到下一次 requester 决策状态。

这样每个 agent 学习的是自身决策序列上的长期收益。

### 4.6 训练伪代码

```text
初始化 PlatformSimulationEnv
初始化 Worker-DQN, Requester-DQN

for episode = 1..N:
    decision = env.reset()
    pending_worker_transition = None
    pending_requester_transition = None

    while decision is not None:
        if decision.actor == worker:
            obs = decision.observation
            若存在 pending_worker_transition:
                用当前 obs 作为 next_obs 写入 worker replay
            action = Worker-DQN.epsilon_greedy(obs)
            step = env.step(action)
            暂存 worker 的 (obs, action, reward)

        if decision.actor == requester:
            obs = decision.observation
            若存在 pending_requester_transition:
                用当前 obs 作为 next_obs 写入 requester replay
            action = Requester-DQN.epsilon_greedy(obs)
            step = env.step(action)
            暂存 requester 的 (obs, action, reward)

        每隔 update_every 步，从对应 replay buffer 采样并更新 Q 网络
        decision = step.next_decision
```

## 5 实验设置

### 5.1 主实验设置

| 参数 | 默认值或说明 |
|------|--------------|
| 数据 | 全量数据，`--max-projects 0` |
| 候选 project 数 | `num_project_candidates=32` |
| 候选 worker 数 | `num_worker_candidates=32`，Requester 动作额外包含 `WAIT` |
| 主结果候选设置 | `include_truth_in_candidates=False` |
| 诊断候选设置 | `--include-truth-in-candidates`，仅用于上限诊断 |
| 模型 | Vanilla DQN / Double DQN / Dueling DQN |
| 优化器 | Adam |
| 损失 | Huber loss |
| 探索 | epsilon-greedy |
| target network | 每 `target_update_freq` 步同步 |

### 5.2 复现命令

```bash
python -m src.dataset --max-projects 0

python scripts/train_platform_dqn.py \
  --max-projects 0 \
  --episodes 20 \
  --max-steps 0 \
  --device cuda \
  --worker-model dueling \
  --requester-model dueling \
  --worker-double-dqn \
  --requester-double-dqn \
  --epsilon-decay-steps 1000 \
  --log-dir runs/report_full_20260529/platform_dqn_full

python scripts/plot_platform_training.py \
  runs/report_full_20260529/platform_dqn_full/platform_dqn_no_truth_20260529_205032/metrics.csv

python scripts/evaluate_platform.py --split test --max-projects 0 \
  --worker-policy dqn --requester-policy dqn \
  --worker-checkpoint runs/report_full_20260529/platform_dqn_full/platform_dqn_no_truth_20260529_205032/checkpoints/worker_best.pt \
  --requester-checkpoint runs/report_full_20260529/platform_dqn_full/platform_dqn_no_truth_20260529_205032/checkpoints/requester_best.pt \
  --output runs/report_full_20260529/platform_dqn_full/platform_dqn_no_truth_20260529_205032/test_eval_best.json

python scripts/run_platform_baselines.py --split test --max-projects 0 \
  --worker-checkpoint runs/report_full_20260529/platform_dqn_full/platform_dqn_no_truth_20260529_205032/checkpoints/worker_best.pt \
  --requester-checkpoint runs/report_full_20260529/platform_dqn_full/platform_dqn_no_truth_20260529_205032/checkpoints/requester_best.pt \
  --output-dir runs/report_full_20260529/platform_baselines_full
```

### 5.3 评价指标

| 指标 | 含义 |
|------|------|
| `worker_hit_rate` | worker 推荐 project 是否对应历史真实投稿 |
| `requester_hit_rate` | requester 选择 worker 是否为历史 winner |
| `worker_reward` | worker 侧累计 reward |
| `requester_reward` | requester 侧累计 reward，不含等待成本 |
| `platform_reward` | worker reward + requester reward - wait cost |
| `project_wait_cost` | project 等候成本 |
| `avg_project_wait_days` | project 平均等待天数 |
| `filled_project_rate` | 成功选出 winner 的 project 比例 |
| `winner_quality` | 被选中 winner 的平均 worker quality |
| `rerouted_workers` | 未中标后被释放并重新推荐的 worker 数 |
| `closed_projects` | 已关闭 project 数 |
| `unfilled_projects` | 到期但未获得候选 worker 的 project 数 |

### 5.4 基线

| 组合 | Worker 策略 | Requester 策略 |
|------|-------------|----------------|
| Random-Wait | random_project | wait_until_deadline |
| Popularity-Quality | popularity | worker_quality |
| Category-Category | category_match | worker_category_match |
| Industry-Industry | industry_match | worker_industry_match |
| Award-Quality | award | worker_quality |
| LowWait-Quality | low_wait_project | worker_quality |
| JointHeuristic | category_match + low_wait_project | worker_quality |
| Platform DQN | DQN | DQN |

## 6 实验结果

数据来源：

- 数据统计与 EDA：`runs/report_full_20260529/data_analysis/`
- 训练曲线：`runs/report_full_20260529/platform_dqn_full/platform_dqn_no_truth_20260529_205032/metrics.csv`
- 图表：`runs/report_full_20260529/platform_dqn_full/platform_dqn_no_truth_20260529_205032/plots/`
- DQN test：`runs/report_full_20260529/platform_dqn_full/platform_dqn_no_truth_20260529_205032/test_eval_best.json`
- 基线对比：`runs/report_full_20260529/platform_baselines_full/platform_test_no_truth/comparison.csv`

### 6.1 训练曲线

需要展示：

- `reward_curves.png`
- `reward_trend_curves.png`
- `hit_rate_curves.png`
- `loss_curves.png`
- `dynamic_state_curves.png`
![dynamic_state_curves.png](../runs/report_full_20260529/platform_dqn_full/platform_dqn_no_truth_20260529_205032/plots/dynamic_state_curves.png)
![epsilon_curves.png](../runs/report_full_20260529/platform_dqn_full/platform_dqn_no_truth_20260529_205032/plots/epsilon_curves.png)
![hit_rate_curves.png](../runs/report_full_20260529/platform_dqn_full/platform_dqn_no_truth_20260529_205032/plots/hit_rate_curves.png)
![loss_curves.png](../runs/report_full_20260529/platform_dqn_full/platform_dqn_no_truth_20260529_205032/plots/loss_curves.png)
![reward_curves.png](../runs/report_full_20260529/platform_dqn_full/platform_dqn_no_truth_20260529_205032/plots/reward_curves.png)
![reward_trend_curves.png](../runs/report_full_20260529/platform_dqn_full/platform_dqn_no_truth_20260529_205032/plots/reward_trend_curves.png)
训练现象：

- 本轮为完整 episode 口径，训练命令使用 `--max-steps 0`，不再做每轮 800 步截断；train 每轮会关闭完整 train split 的 1712 个 project。
- train `platform_reward` 从 episode 1 的 `658.4120` 到 episode 20 的 `633.5106`，最高 episode 15 为 `661.9700`；训练集 reward 受 epsilon 探索、等待成本和回流数量影响，不呈严格单调。
- 验证集最佳 episode 20：`val_platform_reward=242.0326`，`val_worker_hit_rate=0.0574`，`val_requester_hit_rate=0.0000`，`val_project_wait_cost=0.9856`，`val_rerouted_workers=173`。
- episode 10 曾达到 `val_platform_reward=222.0249` 且 `val_requester_hit_rate=0.0074`；episode 20 的总 reward 更高主要来自 worker reward 和回流累计收益，而不是 requester winner 命中提升。
- 因此报告分析时需要同时展示 `platform_reward`、`worker_hit_rate`、`requester_hit_rate`、`project_wait_cost` 与 `rerouted_workers`，不能只看单一累计 reward。

### 6.2 Test 结果表

| 方法 | Worker Hit | Requester Hit | Platform Reward | Wait Cost | Filled Rate | Winner Quality | Rerouted |
|------|------------|---------------|-----------------|-----------|-------------|----------------|----------|
| Random-Wait | 0.0367 | 0.0002 | 913.5873 | 137.9128 | 0.9185 | 0.9012 | 4517 |
| Popularity-Quality | 0.0788 | 0.0136 | 228.1224 | 0.8193 | 1.0000 | 0.8445 | 0 |
| Category-Category | 0.0815 | 0.0109 | 228.7024 | 0.8193 | 1.0000 | 0.8445 | 0 |
| Industry-Industry | 0.0870 | 0.0109 | 231.4816 | 0.8193 | 1.0000 | 0.8445 | 0 |
| Award-Quality | 0.0788 | 0.0109 | 226.4015 | 0.8193 | 1.0000 | 0.8445 | 0 |
| LowWait-Quality | 0.0870 | 0.0109 | 230.6817 | 0.8193 | 1.0000 | 0.8445 | 0 |
| JointHeuristic | 0.0815 | 0.0109 | 228.7024 | 0.8193 | 1.0000 | 0.8445 | 0 |
| Platform DQN | 0.0593 | 0.0070 | 265.1064 | 1.2482 | 1.0000 | 0.8499 | 205 |

### 6.3 结果分析要点

- `requester_hit_rate` 整体较低，说明从申请池中识别历史 winner 仍然困难，winner 信号稀疏；Platform DQN test 只有 `0.0070`。
- `worker_hit_rate` 在无 truth 注入候选集时更接近真实推荐难度，不能与旧实验中强制注入 truth 的 hit rate 直接比较。
- Platform DQN 的 `platform_reward=265.1064` 高于 Popularity、Category、Industry、Award、LowWait、JointHeuristic 等正常启发式组合，主要因为它产生了更多 worker 回流和更高累计 worker reward，同时 winner quality 最高为 `0.8499`。
- `random_project + wait_until_deadline` 的 `platform_reward=913.5873` 异常高，但它伴随 `rerouted_workers=4517`、`unfilled_projects=30`、`project_wait_cost=137.9128`、`steps=10048`。这说明累计 reward 受“等待到 deadline 后大量 worker 回流”放大，不能把该策略简单解释为业务最优。
- Industry/LowWait 的 worker hit 高于 DQN，说明当前 DQN 在 worker 侧匹配能力仍弱于强启发式召回；下一步应改进 worker-project 交互特征或加入监督式排序预训练。

### 6.4 结果可信度与薄弱点

本轮实验能够说明：系统已经完成从“单侧独立推荐”到“动态双边平台仿真”的建模转变。实验不是只在固定候选集上做一次排序，而是在完整时间线上模拟 worker 到达、project 等待、requester 决策、winner 产生和未中标 worker 回流。因此，从作业要求“参与者和发布者动态变化”来看，现有结果具有较强解释力。

但从算法效果看，当前结果还不能证明 Platform DQN 已经全面优于所有基线：

- Worker 侧推荐准确率仍偏弱。Platform DQN 的 `worker_hit_rate=0.0593`，低于 Industry/Industry 和 LowWait/Quality 的 `0.0870`。这说明 DQN 已能在动态环境中运行并学习到一定策略，但在“给 worker 推荐其历史真实会投稿的 project”这一指标上，还没有超过强启发式匹配规则。
- Requester 侧 winner 识别仍是主要短板。Platform DQN 的 `requester_hit_rate=0.0070`，而 Popularity/Quality 为 `0.0136`，多数启发式也在 `0.0109` 左右。原因可能是 winner 信号非常稀疏，且离线日志中只有真实发生的投稿结果，未观察到的 worker-project 组合只能用质量、匹配度和历史分数近似。
- Platform reward 的解释需要谨慎。DQN 的 `platform_reward=265.1064` 高于多数正常启发式，但 Random-Wait 得到 `913.5873` 的异常高值，说明累计 reward 会被“等待到 deadline 后释放大量 worker 并多次回流推荐”放大。因此，`platform_reward` 不能单独作为策略优劣结论，必须同时观察 `worker_hit_rate`、`requester_hit_rate`、`project_wait_cost`、`unfilled_projects` 和 `rerouted_workers`。
- 当前 DQN 的优势更体现在“端到端动态机制可运行”和“多目标指标统一记录”，而不是在所有单项准确率上取得最优。因此报告结论应表述为：Platform DQN 为动态双边众包推荐提供了可训练框架，并在平台收益与项目填充方面表现稳定，但 worker 侧匹配精度和 requester 侧 winner 选择仍需进一步优化。

面向老师展示时，建议强调两层结论：第一，本文完成了符合动态参与者/发布者要求的联合仿真环境和双智能体 DQN 建模；第二，当前实验结果揭示了 reward 尺度、稀疏 winner 信号和候选召回质量对策略效果的影响，因此后续工作会围绕 reward 归一化、监督式预训练和 requester 侧辅助学习继续改进。

## 7 讨论

### 7.1 动态性回应

- project 的 active/closed/deadline 状态随时间变化。
- worker 会因进入申请池而暂时不可被推荐。
- requester 可以等待更多申请者，而不是必须立即选择。
- project 关闭后未中标 worker 会回流，平台形成连续推荐过程。

### 7.2 离线仿真的局限

- 历史日志只记录实际发生的 worker-project 投稿，未观察到的组合 reward 只能用质量、匹配度、奖金等 proxy 构造。
- winner/finalist 是稀疏信号，Requester-DQN 学习难度较高。
- 当前候选召回仍是启发式 top-K，DQN 主要学习候选集内排序和等待/选择决策。
- `include_truth_in_candidates=True` 只能作为诊断上限，主结果应使用默认 `False`。
- 累计式 `platform_reward` 对回流次数敏感；当策略造成大量 worker 在 deadline 后被释放并重复推荐时，总 reward 可能被放大，因此需要配合平均等待时间、未填充项目数和命中率共同解释。
- 当前单次全量训练尚未覆盖多随机种子稳定性分析；因此结果更适合说明建模框架和一次正式实验现象，不能过度表述为稳定显著优于所有基线。

### 7.3 可改进方向

- 对 worker-project 进行监督式排序预训练，再用 DQN 微调。
- 对 requester 增加 winner 识别辅助损失。
- 对 platform reward 做按 project 或按决策步归一化，减少回流次数对累计 reward 的尺度影响。
- 增加申请池质量分布特征，使 requester 判断“是否继续等待”更有信息。
- 引入更严格的离线策略评估方法，如 IPS / Doubly Robust。

## 8 结论

- 本文将众包任务推荐建模为动态双边平台 MDP，克服了旧独立环境无法表达 worker 与 project 动态变化的问题。
- Worker-DQN 学习“当前 worker 应推荐哪个 project”，Requester-DQN 学习“当前 project 应继续等待还是选择 winner”。
- 候选集 Q 网络能够处理每一步候选 project/worker 集合变化的问题。
- 平台 reward 将参与者收益、请求者收益与 project 等待成本统一到同一评价框架。
- 全量实验表明，Platform DQN 能够在动态联合环境中完成端到端训练，并在项目填充率和多数正常启发式对比的 platform reward 上表现稳定。
- 但当前 DQN 尚未在 worker hit 和 requester hit 上全面超过强启发式基线，且累计 reward 会受到 worker 回流次数影响。因此本文更稳妥的结论是：动态双边建模框架已经成立，算法效果仍需通过更好的特征、reward 归一化和 requester 侧辅助学习继续增强。

---

## 附录

### A 小组成员分工

| 成员 | 负责内容 |
|------|----------|
| | 数据、特征 |
| | 环境、MDP |
| | 模型、训练 |
| | 实验、报告 |

### B 关键超参表

（从 `runs/.../config.json` 粘贴）

### C 额外实验截图

---

## 汇报 PPT 建议结构（10–15 分钟）

1. 问题与数据（2 min）
2. MDP 建模示意图（3 min）
3. 网络与训练（2 min）
4. 实验结果表 + 曲线（4 min）
5. 结论与 Q&A（2 min）
