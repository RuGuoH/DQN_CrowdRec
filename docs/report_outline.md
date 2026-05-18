# 强化学习大作业实验报告大纲

> 课程：强化学习  
> 题目：基于 DQN 的众包任务推荐  
> 小组：____（4人）  
> 日期：2026年6月  

---

## 摘要（200–300字）

- 研究问题：众包平台如何向参与者推荐任务 / 向请求者匹配 worker
- 方法：离线历史日志 + MDP 建模 + DQN 系列算法
- 主要结论：Hit@1、累计 reward；与基线对比的提升幅度
- 关键词：众包、任务推荐、深度强化学习、DQN

---

## 1 引言

### 1.1 背景

- 众包平台（Crowdspring）业务流程
- Amazon MTurk 式「发布任务—参与者选择—完成聚合」
- 平台需同时兼顾参与者收益与请求者稿件质量

### 1.2 问题定义

- **简化假设**：每次仅推荐 **1** 个任务（参与者侧）或 **1** 个 worker（请求者侧）
- 两个优化目标：
  1. 最大化参与者利益（相关任务、更高收益）
  2. 最大化请求者利益（更多高质量投稿）

### 1.3 本文工作

- 构建双端 MDP 与特征工程
- 实现 DQN / Double DQN / Dueling DQN
- 与多种基线对比，给出实验分析与结论

---

## 2 数据分析

### 2.1 数据来源

- Crowdspring 数据集字段说明（`project_list.csv`、`worker_quality.csv`、`project/`、`entry/`）
- 时间范围、过滤条件（`min_start_date >= 2018-01-01`）

### 2.2 描述性统计（需配图/表）

| 统计项 | 数值 |
|--------|------|
| 项目数 | |
| 投稿条目数 | |
| Worker 数（有质量分） | |
| 平均每项目投稿数 | |
| 行业/类目分布 | |

**建议图表**：

- 项目 `start_date` 时间分布直方图
- `entry_count` / `total_awards` 分布
- Worker `quality` 分布

### 2.3 训练/验证/测试划分

- **按项目 `start_date` 排序** 后切分：70% / 15% / 15%
- 说明：避免时间泄漏，符合「用过去预测未来」

---

## 3 方法

### 3.1 总体框架

```
历史日志 → 事件流 → MDP 环境 → DQN → 推荐动作
                ↘ 基线对比
```

### 3.2 参与者侧 MDP（对应作业问题 1）

| 要素 | 定义 |
|------|------|
| 状态 \(s_t\) | worker 特征 \(\mathbf{w}\) + K 个候选项目特征矩阵 \(\mathbf{C}\) + mask |
| 动作 \(a_t\) | \(a \in \{0,\ldots,K-1\}\)，选择推荐项目 |
| 奖励 \(r_t\) | 命中真实投稿 \(+\,R_{\text{hit}}\)；加分：revision score、winner、finalist；未命中惩罚 |
| 转移 | 推进至下一条 worker 投稿事件 |

**特征向量（写明维度）**：

- Worker（8维）：quality、历史投稿数、均分、胜率、决赛率、主导类目、距上次投稿间隔等
- Project（10维）：类目、行业、投稿数、奖金、均分、featured、剩余时间等

### 3.3 请求者侧 MDP（对应作业问题 2）

| 要素 | 定义 |
|------|------|
| 状态 | 项目上下文（10维）+ K 个 worker 特征（各8维） |
| 动作 | 推荐 1 名 worker |
| 奖励 | 命中真实投稿 worker + 质量分 + 分数/获奖加成 |

### 3.4 Q 函数与网络结构

- \(Q(s,a;\theta)\)：anchor MLP + candidate MLP 拼接后输出 K 维 Q 值
- **Vanilla DQN**、**Double DQN**（解耦 argmax 与评估）、**Dueling DQN**（\(Q=V+A-\text{mean}(A)\)）
- 非法动作 mask 为 \(-\infty\)

### 3.5 训练细节

| 超参数 | 取值 |
|--------|------|
| K（候选数） | 32 |
| \(\gamma\) | 0.99 |
| 学习率 | 1e-3 |
| batch size | 32/64 |
| replay size | 10000 |
| \(\epsilon\) 衰减 | 1.0 → 0.05 |
| target 更新频率 | 每 200 步 |

### 3.6 基线方法

**参与者侧**：

| 基线 | 策略 |
|------|------|
| Random | 合法动作均匀随机 |
| Popularity | 选投稿数最多项目 |
| CategoryMatch | 选与 worker 主导类目一致项目 |
| Award | 选奖金最高项目 |

**请求者侧**：

| 基线 | 策略 |
|------|------|
| Random | 随机 worker |
| WorkerQuality | 选质量分最高 worker |
| WorkerActivity | 选历史投稿最多 worker |

---

## 4 实验

### 4.1 实验环境

- Python 版本、PyTorch 版本、硬件（CPU/GPU）
- 代码仓库结构与复现命令（见 README）

### 4.2 复现命令（填入实际路径）

```bash
# 数据检查
python -m src.dataset --max-projects 0

# 训练
python scripts/train_worker_dqn.py --max-projects 0 --episodes 20
python scripts/train_requester_dqn.py --max-projects 0 --episodes 20

# 基线 + 对比
python scripts/run_baselines.py --side worker --split test --max-projects 0
python scripts/run_baselines.py --side requester --split test --max-projects 0

# 单模型评估
python scripts/evaluate.py --side worker --split test --policy dqn \
  --checkpoint runs/worker/.../checkpoints/best.pt
```

### 4.3 评价指标

- **Hit@1**：推荐是否命中真实 project/worker
- **Average Reward**：episode 累计奖励 / 步数
- （可选）NDCG@K、MRR（若扩展多候选排序）

### 4.4 主实验结果（表 1：test 集）

**参与者侧**

| 方法 | Hit@1 | Avg Reward | 备注 |
|------|-------|------------|------|
| Random | | | |
| Popularity | | | |
| CategoryMatch | | | |
| Award | | | |
| DQN | | | |
| Double DQN | | | |
| Dueling DQN | | | |

**请求者侧**

| 方法 | Hit@1 | Avg Reward | 备注 |
|------|-------|------------|------|
| Random | | | |
| WorkerQuality | | | |
| WorkerActivity | | | |
| DQN | | | |

> 数据来源：`runs/baselines/{side}_test/comparison.csv`

### 4.5 学习曲线

- 贴 `runs/.../metrics.csv` 绘制的 train/val reward、hit_rate、loss 曲线
- 分析：是否过拟合、\(\epsilon\) 衰减是否合理

### 4.6 消融实验（可选）

- K = 8 / 16 / 32 对 Hit@1 的影响
- 是否将真实标签放入候选集（`include_truth_in_candidates`）
- reward 权重敏感性

### 4.7 案例分析（1–2 个）

- 成功推荐：worker 类目与项目匹配
- 失败案例：开放项目过少、候选未覆盖真实标签

> **注意**：若候选集始终包含真实标签（`include_truth_in_candidates=True`），CategoryMatch 等启发式可能接近 Hit@1=1，需在报告中说明该设定并补充「不含 truth 的候选集」消融。

---

## 5 讨论

### 5.1 参与者 vs 请求者目标

- 两套模型独立优化，目标可能冲突
- 未来：多目标 RL 或加权单目标

### 5.2 离线强化学习的局限

- 分布偏移：行为策略 ≠ 学习策略
- 可考虑：BC 预训练 + DQN 微调

### 5.3 与作业假设的对应

- 「每次只推荐 1 个任务」如何落实
- 请求者侧「推荐 worker」的对称解释是否合理

---

## 6 结论

- 总结主要发现（3–4 条）
- 说明 DQN 是否显著优于基线
- 简要展望

---

## 参考文献

1. Mnih et al., Human-level control through deep reinforcement learning, 2015.
2. van Hasselt et al., Deep Reinforcement Learning with Double Q-learning, 2016.
3. Wang et al., Dueling Network Architectures for Deep Reinforcement Learning, 2016.

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
