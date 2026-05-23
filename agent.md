# Agent 协作指南（强化学习大作业 · 众包任务推荐）

> **读者**：Cursor / 其它 AI Agent、使用 AI 改代码的同学。  
> **用途**：统一项目目标、实现现状、修改边界与协作方式。人类用户请同时参考 `README.md` 与 `docs/report_outline.md`。

---

## 1. 项目目标（不可偏离）

### 1.1 课程作业要求

| 项 | 内容 |
|----|------|
| 场景 | Crowdspring 众包平台历史日志；**每次只推荐 1 个对象** |
| 问题 1 | 强化学习做**任务推荐**，最大化**参与者（worker）**利益 |
| 问题 2 | 强化学习做推荐，最大化**请求者（project 发布方）**利益 |
| 方法 | **必须使用 DQN 系列**（Vanilla / Double / Dueling） |
| 数据 | `data/data/`；自行划分 train/val/test；可参考 `sample_read_data.py` |
| 交付 | 实验报告（流程、设计、结果）+ 分组汇报；截止以课程通知为准 |

### 1.2 本仓库的实现取向

- **离线强化学习**：用历史 `(state, action, reward, next_state)` 事件流仿真，非在线 API 环境。
- **双端独立 MDP**：参与者侧、请求者侧各一套环境 + 各训一个 DQN（非多目标单模型）。
- **工程目标**：可复现训练 → 评估 → 基线对比 → 填报告表格。

---

## 2. 仓库结构（修改前先读）

```
强化学习/
├── agent.md                 # 本文件（AI 协作入口）
├── README.md                # 人类快速上手
├── configs/default.yaml     # 数据路径、划分比例、DQN 默认超参
├── data/data/               # 原始数据（勿改内容；只读）
│   ├── project_list.csv
│   ├── worker_quality.csv
│   ├── project/project_{id}.txt
│   └── entry/entry_{id}_{offset}.txt
├── src/
│   ├── config.py            # 加载 YAML
│   ├── dataset.py           # 数据加载、划分、事件流、cache v2
│   └── features.py          # Worker(8维) / Project(10维) 特征
├── env/
│   ├── worker_env.py        # 参与者 MDP
│   └── requester_env.py     # 请求者 MDP
├── models/
│   ├── dqn.py               # Q 网络、DQNAgent、checkpoint
│   ├── baselines.py         # 基线策略
│   ├── eval_utils.py        # 评估循环
│   ├── eval_runner.py       # evaluate_one（脚本共用）
│   ├── train_utils.py       # 训练 episode 循环
│   └── training_log.py      # CSV/JSON 日志
├── scripts/
│   ├── train_worker_dqn.py
│   ├── train_requester_dqn.py
│   ├── evaluate.py          # 单策略评估
│   ├── run_baselines.py     # 批量基线 + 可选 DQN
│   ├── smoke_env.py
│   └── smoke_requester.py
├── docs/report_outline.md   # 实验报告大纲（填结果用）
├── cache/                   # dataset_*.pkl（自动生成，可删后重建）
└── runs/                    # 训练/评估输出（勿提交超大 checkpoint 除非课程要求）
```

### 2.1 关键数据约定

- Entry JSON 中 worker 字段名为 **`author`**（不是 `worker`）；`dataset.py` 已处理。
- `project_list.csv`：`project_id, entry_count`；entry 文件分页偏移 0,24,48,...
- 划分：**按项目 `start_date` 排序** 后 70% / 15% / 15% → train/val/test（在 `dataset.py`）。
- 过滤：`start_date >= 2018-01-01`（与 `sample_read_data.py` 一致）。

### 2.2 观测张量约定（DQN 输入）

`env.worker_env.Observation` 被两侧复用，**字段名不变、语义不同**：

| 字段 | 参与者侧 | 请求者侧 |
|------|----------|----------|
| `worker_feat` | worker 特征 (8) | **项目上下文** (10) |
| `candidate_feat` | K 个项目特征 (K×10) | K 个 worker 特征 (K×8) |
| `action_mask` | 合法候选槽位 | 同上 |

请求者侧训练时：`DQNConfig(anchor_dim=10, candidate_dim=8)`；参与者侧：`anchor_dim=8, candidate_dim=10`。

---

## 3. 已实现 vs 未完成

### 3.1 已完成（勿重复造轮子）

- [x] 数据管道 `CrowdsourcingDataset` + pickle 缓存 v2
- [x] `iter_worker_events` / 活跃项目查询
- [x] 参与者 MDP `WorkerRecommendationEnv`
- [x] 请求者 MDP `RequesterRecommendationEnv`
- [x] DQN / Double / Dueling + Replay + target 网络
- [x] 训练日志 `metrics.csv`、`config.json`、checkpoint（best/ep/final）
- [x] 基线：`random`, `popularity`, `category_match`, `award`（worker）；`worker_quality`, `worker_activity`（requester）
- [x] `evaluate.py`、`run_baselines.py`
- [x] 报告大纲 `docs/report_outline.md`
- [x] `include_truth_in_candidates` 消融（train/eval/baseline 已支持 CLI 开关）

### 3.2 未完成（优先任务）

- [ ] **全量数据**正式实验（`--max-projects 0`，足够 episode）
- [ ] 三种 DQN 变体系统对比并填入报告表
- [ ] **实验报告正文**（PDF/Word）与 **PPT**
- [ ] 数据分析 EDA 图表写入报告 §2
- [ ] 学习曲线出图脚本（可从 `metrics.csv` 绘制）
- [ ] 可选：BC 预训练、TensorBoard、GPU 默认配置

---

## 4. AI Agent 行为标准

### 4.1 通用原则

1. **先读后改**：修改前阅读 `agent.md` → 相关模块 → 调用方脚本；不要假设 API。
2. **小步提交**：单次 PR/任务只解决一个明确问题（如一侧环境、一脚本、一表）。
3. **可运行验证**：改完后至少运行相关 smoke 或 `evaluate`；全量训练由用户触发。
4. **不破坏数据**：不修改 `data/data/` 内原始文件；不提交 `cache/`、`runs/` 大文件除非用户要求。
5. **中文注释适度**：公开 API 与复杂逻辑用简短中文/英文均可；避免冗长注释。
6. **匹配现有风格**：dataclass 配置、`build_dataset()`、`Observation.to_dict()` 等沿用现有模式。

### 4.2 禁止事项

- 勿删除或重命名 `runs/` 下用户实验结果（除非用户明确要求清理）。
- 勿将 `test` 集用于调参或 early stopping（仅最终报告）。
- 勿在 `_build_candidates` 中用「随机填充至 K」死循环逻辑（历史 bug，已修复）。
- 勿用 `worker` 字段读 entry（应用 `author`）。
- 勿引入与作业无关的大型依赖（如 Ray、完整 RLlib）除非用户明确要求。
- 勿擅自写长篇 `.md`（除 `agent.md`、`report_outline` 和用户点名要的文档）。

### 4.3 修改范围指南

| 你想做的事 | 应改文件 | 避免改 |
|------------|----------|--------|
| 特征工程 | `src/features.py` | 直接改 JSON 数据 |
| 参与者奖励/候选 | `env/worker_env.py` | `models/dqn.py` 网络结构（除非必要） |
| 请求者奖励/候选 | `env/requester_env.py` | `worker_env.py` |
| 网络/算法 | `models/dqn.py` | 环境 step 逻辑混杂进 Agent |
| 训练流程 | `scripts/train_*.py`, `models/train_utils.py` | 复制粘贴成第三套训练脚本 |
| 评估/基线 | `models/baselines.py`, `models/eval_runner.py`, `scripts/evaluate.py` | 重写 dataset |
| 超参默认值 | `configs/default.yaml` + 脚本 argparse | 硬编码在多处 |

### 4.4 测试命令（改代码后）

```bash
# 最快：环境 + 网络前向
python scripts/smoke_env.py
python scripts/smoke_requester.py

# 数据
python -m src.dataset --max-projects 50

# 短训练
python scripts/train_worker_dqn.py --max-projects 50 --episodes 2 --max-steps 200

# 评估
python scripts/run_baselines.py --side worker --split test --max-projects 50
```

---

## 5. 已知陷阱与领域知识

### 5.1 `include_truth_in_candidates=True`（默认）

训练与评估时，**真实标签几乎总在 K 个候选里**。会导致：

- `category_match` 等基线 Hit@1 **虚高**（子集实验上可达 ~1.0）。
- DQN 的 Hit@1 相对基线优势可能被低估或对比失真。

-(此行删除)**Agent 任务**：若做公平对比，应实现 CLI `--no-truth-in-candidates` 并跑消融；报告中必须说明两种设定。
- 训练与评估时，默认会将真实标签强制加入 K 个候选中：

```python
include_truth_in_candidates=True

### 5.2 Cache

- 路径：`cache/dataset_{all|n50}.pkl`，version=2 字典序列化。
- 从脚本运行 pickle 安全；缓存损坏时会自动全量重解析。
- 换 `max_projects` 会生成不同 cache 文件。

### 5.3 Entry 字段

- `revisions[].score`、`winner`、`finalist` 用于奖励。
- `withdrawn=True` 的 entry 不进入事件流。

### 5.4 性能

- 全量事件上万步；应用 `max_steps` 做调试，正式实验去掉步数上限。
- 特征已用 `bisect` 优化历史查询；避免在 `step()` 里全表扫描。

---

## 6. 多 Agent 协作分工建议

| 角色 | 职责 | 主要文件 | 交接物 |
|------|------|----------|--------|
| **Data** | EDA、划分说明、特征文档 | `dataset.py`, `features.py`, 报告 §2 | 统计表、图 |
| **Env** | MDP 定义、奖励、候选逻辑 | `env/*.py`, 报告 §3.2–3.3 | 公式与参数表 |
| **RL** | DQN 变体、训练、调参 | `models/dqn.py`, `train_*.py` | `runs/*/checkpoints/best.pt` |
| **Eval** | 基线、test 评估、填表 | `baselines.py`, `run_baselines.py` | `comparison.csv` |
| **Report** | 正文、PPT、可复现命令 | `docs/report_outline.md` | PDF + 命令清单 |

**协作规则**：

1. 每 Agent 开工前读取 `agent.md` 与 `runs/*/config.json`（如有）。
2. 改接口（Observation 字段、checkpoint 格式）必须在 `agent.md` 或 PR 说明中**显式通知**其它角色。
3. 实验结果只认 **`split=test`** 且写入 `runs/baselines/*_test/comparison.csv` 的行。

---

## 7. 推荐提示词（复制给 AI）

### 7.1 新 Agent 入门

```
你正在参与「强化学习大作业 · 众包任务推荐」项目。
请先阅读仓库根目录 agent.md、README.md，再读你要改的文件。
目标：课程要求的双端 DQN 推荐；不要偏离 offline RL + 双 MDP 架构。
改完后运行 agent.md §4.4 中最相关的 smoke/评估命令并汇报结果。
```

### 7.2 跑全量实验

```
在 agent.md 约束下，使用全量数据（--max-projects 0）完成：
1) train_worker_dqn 与 train_requester_dqn（episodes≥15）；
2) test 集 run_baselines（含 best.pt）；
3) 将 comparison.csv 关键指标总结为 Markdown 表格。
不要改动 data/data/；记录完整命令与 runs 路径。
```

### 7.3已删除

```

### 7.4 写报告某节

```
根据 docs/report_outline.md 第 N 节提纲撰写正文。
引用仓库已实现公式与文件路径；实验数字只使用 runs/ 下真实 CSV/JSON，缺失处写「待填」勿编造。
中文，学术语气，篇幅与提纲匹配。
```

### 7.5 Debug 训练慢/卡住

```
阅读 agent.md §5.4 与 worker_env._build_candidates。
用 python -u 对 step() 做 50 步计时；检查 while 死循环、特征全量扫描、错误 cache。
修复后更新 agent.md 已知陷阱一节。
```

---

## 8. 全量实验前检查清单（Agent 可代为执行）

```
□ python -m src.dataset --max-projects 0  # 确认能加载
□ smoke_env + smoke_requester 通过
□ 确定 episodes、K、seed、DQN 变体列表
□ 决定 include_truth 是否做消融
□ train worker → train requester → run_baselines --checkpoint best.pt
□ test 集结果写入 report_outline §4.4 表格
□ 从 metrics.csv 出学习曲线图
```

---

## 9. 文档与入口索引

| 文档 | 受众 |
|------|------|
| `agent.md`（本文件） | AI Agent、协作规范 |
| `README.md` | 人类快速命令 |
| `docs/report_outline.md` | 实验报告结构 |

**人类同学**：改代码用 AI 时，在对话首条 @ 本文件或粘贴 §7 提示词。  
**AI Agent**：把本文件当作 Source of Truth；与 `README` 冲突时，以**作业要求**和**本文件 §1、§3** 为准。

---

*最后更新：与仓库实现同步（含 evaluate、baselines、双端 env、DQN 日志与 checkpoint）。*
