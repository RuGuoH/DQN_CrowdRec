# 平台双边 DQN 设计讨论纪要

> 记录开放讨论中有价值的结论与后续优化方向。  
> 涵盖：模型规模、学习率、utility 与历史 hit 的关系、双边推荐机制、Worker 候选召回、「多次落选」体验问题、历史 baseline 与仿真对比、项目异质诉求（质量 vs 速度）、Reward / platform_reward 设计问题，以及 **§13 待进一步考虑的方向**。

---

## 1. 项目设计在说什么

### 1.1 核心机制（Run B / `reward_mode=utility`）

平台是**异步双 Agent MDP**：

- **Worker-DQN**：在 Top-K 项目候选里，选 **U_worker 更高** 的项目去申请。
- **Requester-DQN**：在申请池召回的 Top-K 工人（或 **WAIT**）里，选 **U_requester 更高** 的 winner；等待会产生 `wait_cost`。

可以概括为：**「双边各自做效用最大化推荐」**，但不是全库全局最优，而是**在各自可见候选集内**按手工定义的 utility proxy 决策。

### 1.2 Worker Utility（训练主目标）

\[
U^W = 0.35 \cdot \text{award} + 0.25 \cdot \text{match}_{cat/ind} + 0.25 \cdot \text{skill} - 0.15 \cdot \text{competition}
\]

- `award`：项目奖金（log 归一化）
- `match`：Worker 主导类目/行业与 Project 是否一致
- `skill`：该类目历史均分与胜率
- `competition`：项目已有投稿量（竞争压力）

命中历史 truth 时额外 +0.1（`legacy_hit_weight`），相对 utility 量级很小。

### 1.3 Requester Utility

\[
U^R = 0.40 \cdot Q_{worker} + 0.35 \cdot \mathbb{E}[\text{score}] + 0.15 \cdot \text{match} + 0.10 \cdot \text{activity}
\]

实际 Agent reward 还要减去等待成本。Requester 学的是 **「现在选」vs「再等等」** 的权衡，不是无脑选 utility 最高的人。

### 1.4 三个关键限定

| 限定 | 说明 |
|------|------|
| **候选集内最优** | Worker 只见 Top-32 项目；Requester 只见申请池召回的工人。Recall 错了，排序救不了。 |
| **各自利益，非平台总福利** | 两个独立 Agent，没有统一的 social welfare 联合优化。 |
| **Utility 是 proxy** | 与真实福利、历史行为、平台长期指标只有部分相关。 |

---

## 2. Worker Top-K 候选如何产生

每次 Worker 决策时，`_build_project_candidates` 动态生成 **K=32** 个槽位（Run B 默认）。

### 2.1 第一步：圈定 active 池

保留同时满足：

- 项目未关闭，且 `t < deadline`
- `start_date ≤ t + lookahead`（默认 **168h / 7 天**，含即将开放项目）
- Worker **尚未申请**过该项目
- 默认 **不**强制注入 truth（`include_truth_in_candidates=False`）

### 2.2 第二步：Mixed Recall（Run B 默认）

从 active 池用**四路混合召回**凑满 32 个（去重）：

| 通道 | 约占比 | 规则 |
|------|--------|------|
| 匹配 | ~50% | 优先 Worker 主导类目/行业一致，再按等待天数、投稿量 |
| 热门 | ~17% | 投稿量、奖金高 |
| 低等待 | ~17% | 平台侧等待天数长（等项目较久） |
| 随机 + 兜底 | 余量 | 探索性补齐 |

若关闭 mixed recall（`--no-mixed-recall`），则改为按等待天数 → 投稿量 → 奖金单排序截断。

### 2.3 第三步：DQN 只做候选内排序

候选定好后编码为 `candidate_feat[K×14]`，Worker-DQN 在 32 槽位里 argmax Q。**RL 不改召回规则，只改排序。**

### 2.4 与 Requester 候选对比

| | Worker | Requester |
|--|--------|-----------|
| 来源 | 全局 active 项目 | **当前项目申请池** |
| 召回 | mixed recall 四路 | 历史 winner 优先 + 质量/活跃度 |

---

## 3. Utility 与历史 Hit：两套不同的「好」

### 3.1 定义

| 概念 | 问的是什么 | 信息集 |
|------|------------|--------|
| **Utility 最大化** | 按我们定义的规则，当时哪个选择更划算？ | 决策时刻可观测特征 |
| **历史 Hit** | 策略选择是否等于日志里的真实结果？ | 后验全知（ground truth） |

- **Worker hit**：所选 project 是否为该 Worker 历史上真实投稿的项目（`submitted` 或 `truth_project_id` 一致）。
- **Requester hit**：所选 worker 是否为该项目历史 **winner**。

### 3.2 在项目里的角色

| 角色 | Run B 中的用法 |
|------|----------------|
| **训练主目标** | Utility（hit 仅 +0.1 弱 bonus） |
| **BC 标签** | Utility 模式下 argmax U，不是 argmax hit |
| **Checkpoint 选择** | 以 utility 为主（`U_w + 5·U_r + 弱 hit/recall`） |
| **Hit 指标** | **离线评估 / 诊断**，衡量能否复现历史行为 |
| **Legacy 模式** | `reward_mode=legacy` 时以 hit 为主（对照实验） |

### 3.3 为什么二者不必一致

1. **Utility 是手工 proxy**，不等于真人真实动机（时间、习惯、信息不全等）。
2. **Hit 假设「历史 ≈ 最优」**，在众包/marketplace 里常不成立。
3. **Hit 极稀疏**（尤其 Requester winner），utility 会选对「高质量一类人」，但未必命中唯一 winner。
4. **特征装不下决定因素**，Recall 高、Hit 低 = 看见正确答案但排序不对。

### 3.4 三种「好」的框架（开放讨论）

| 类型 | 含义 | 典型指标 |
|------|------|----------|
| **描述性** | 人们实际怎么做 | 历史 hit、log-likelihood |
| **规范性** | 按我们的标准应该怎样 | Utility、social welfare |
| **因果/反事实** | 干预后是否真的更好 | 在线 A/B、IPS/DR |

**不要默认其中一种可以替代另一种。** Run B 的现象（utility 最高、worker hit 最低、platform 并非最优）正是这种错位的体现。

---

## 4. 模型与学习率（简要结论）

### 4.1 MLP 规模

- 结构：Anchor–Candidate 双塔，**Dueling**，`hidden_dim=128`
- 参数量：单网络约 6–8 万，双 Agent 合计约 **13–16 万**
- **结论**：对 12–17 维手工特征 + Top-32 排序，**合理且偏保守**；不是当前瓶颈。主要限制在探索、reward 对齐、训练预算。

### 4.2 学习率

| 阶段 | lr | 说明 |
|------|-----|------|
| Platform DQN | **3×10⁻⁴** | Adam，Huber loss，grad clip=10 |
| BC 预训练 | **1×10⁻³** | 载入权重后 optimizer 重置 |
| `default.yaml` | 1×10⁻³ | **不被** `train_platform_dqn.py` 读取 |

- **结论**：3×10⁻⁴ 对 Double Dueling DQN 是常见安全默认；BC→RL 降 lr 合理。
- Requester loss 更抖、wait 不稳定时，可试 **`--requester-lr 1e-4`**（比全局加大网络更值得先试）。

---

## 5. 多次投递却屡落选：体验问题与优化建议

### 5.1 问题

Worker 连续申请多个项目却未中标，会：

- 浪费等待期（申请后 `busy`，直到项目关闭才 reroute）
- 产生挫败感，不符合其长期利益
- 在平台上可能被反复推「高诱惑、低胜率」的热门项目

**当前设计**：落选后会 `rerouted_workers` 释放回队列，但 Worker utility **未显式建模**落选 streak、等待机会成本、期望胜率。

### 5.2 优化方向（按改动成本排序）

#### A. Worker Reward / Utility（RL 最直接）

1. **落选惩罚**：在 `_close_project` 对非 winner 发 delayed negative reward。
2. **等待机会成本**：utility 减去 `λ × 预期剩余等待天数`。
3. **Streak 特征与惩罚**：特征中加入 `recent_applies / recent_wins / consecutive_rejections`；对高竞争 + 低近期胜率的项目再降权。
4. **校准期望收益**：

   \[
   U^W_{\text{calibrated}} = P(\text{win} \mid w,p) \cdot V(p) - (1-P) \cdot C_{\text{reject}}
   \]

   避免「奖金高但几乎不可能中」的项目长期霸榜。

#### B. 召回策略（减少「看得见摸不着」）

- 对连输 Worker：**降低「热门」通道权重**，提高「低竞争 / 高匹配 / 小池子」占比。
- Win-rate-aware 过滤：估计 `P(win)` 低于阈值的项目不进 Top-K。
- 同类项目连续落选 → 短期冷却（cooldown）。

#### C. Requester / 平台机制（双边公平）

- Requester utility 加 **fairness 项**（长期未中标但质量尚可的 worker 适度加分，需防刷分）。
- 平台约束：滑动窗口内最低曝光、限制同时 pending 项目数。
- 与 wait/batch 机制联动，减少「投进去池子太小、根本没机会」的情况。

#### D. 评估指标（否则改了也看不见）

| 指标 | 用途 |
|------|------|
| Per-worker win rate / apply count | 是否有人被系统性牺牲 |
| Rejection streak 分布 | 公平性 |
| Time-to-first-win | 新 Worker 冷启动体验 |
| Rejected-after-wait rate | 白等后落选比例 |

平台目标可考虑：

\[
\max \sum_w U_w - \beta \cdot \text{Inequality} - \gamma \cdot \text{AvgRejectionStreak}
\]

### 5.3 与本项目 RL 框架的衔接

| 优先级 | 改动 | 文件/位置 |
|--------|------|-----------|
| 低 | Worker 特征加 streak 统计；utility 加 streak / 竞争惩罚 | `src/features.py`, `env/platform_env.py` |
| 中 | 落选 delayed reward；mixed recall 对连输 worker 减热门权重 | `env/platform_env.py` |
| 高 | Requester fairness；约束 RL / 多目标 checkpoint | `env/platform_env.py`, `scripts/train_platform_dqn.py` |

BC 标签也可从「静态 utility 最大」改为「**校准后期望收益最大**」。

---

## 6. Run B 实验与上述讨论的对照

| 现象 | 讨论中的解释 |
|------|--------------|
| Test utility 合计最高（0.977） | Proxy 学好了，不代表历史复现或平台最优 |
| Worker hit 最低（5.56%），recall@k 很高（0.94） | 候选内排序问题，不是看不见 truth |
| Platform reward 低于 industry_match | 双边各自 utility 最优 ≠ 平台运营最优 |
| Wait 高、5 个 unfilled | Requester WAIT 边界不稳定；utility 未惩罚 unfilled |
| ep10 后 utility 回落 | 探索不足（Worker ε ep1 触底）、训练步数有限，非 MLP/lr 主因 |
| 训练约 25 分钟 | 小网络 + GPU + 20 ep + update_every=4，合理但学习不充分 |

---

## 7. 开放问题（待进一步讨论）

1. **Utility 权重**是否应通过 val 上 platform/wait/fill 联合调参，而非只盯 utility？
2. **Worker 探索**：BC 后是否应重置 ε，或延长 `epsilon_decay_steps`（如 15000）？
3. **Checkpoint 公式**是否应加入 `wait_cost`、`filled_rate`、fairness 约束？
4. **Hit 指标**在报告中应明确标注为「离线复现度」，与 utility「规范目标」分开展示。
5. **Apply→Win**（仿真策略下的申请胜率）与**历史 hit**是不同概念，评估时需区分。
6. 是否实现 `scripts/historical_baseline.py`，对 test split 输出历史 utility / apply-win baseline？
7. **项目 tier**（快选 vs 慢选）规则如何定？是否用 featured / 奖金 / deadline 压力自动分档？
8. **platform_reward** 是否改为主报 `platform_reward_per_project`，并补 unfilled / 落选项？
9. Reward 改版后，checkpoint 与报告主表字段如何统一？

更完整的待办清单见 **§13**。

---

## 8. 历史 baseline：能否从日志算 utility 与申请胜率？

### 8.1 项目定义的 utility，历史里能算吗？

**能**，前提：用与仿真**同一套公式**、**同一时刻**的可观测特征，且不算未发生的反事实。

Run B 的 \(U^W\)、\(U^R\) 在 `platform_env.py` 中由决策时刻 `t` **之前**的历史统计 + 项目/Worker 属性计算，不偷看该 (worker, project) 的未来 outcome。因此可对日志里**真实发生**的 (worker, project, 时间) 逐条回放。

#### Worker utility（历史可算）

对每条真实投稿（`EntryRecord`，非 withdrawn）：

- 时间：`entry_created_at`
- 主体：`worker_id`，`project_id`
- 调用与仿真相同的 `_compute_worker_utility(worker_id, project_id, t)`

聚合：`mean(U^W)` over all entries；可按 worker / 类目 / split 分组。

#### Requester utility（历史可算，时间口径需固定）

对每个产生 winner 的项目：

- 主体：历史 winner `(project_id, winner_worker_id)`
- 时间：可用 winner 的 `entry_created_at`，或项目 deadline 前某固定规则（全文统一即可）
- 调用 `_compute_requester_utility(project_id, winner_id, t)`

#### 重要限定：这是 proxy，不是真实收入

| 能反映 | 不能反映 |
|--------|----------|
| 公式中的奖金、匹配、skill、质量等 | 真实到账、满意度、长期声誉 |
| 与仿真 `avg_worker/requester_utility` **同口径**的可比量 | 未申请 (worker, project) 的反事实效用 |

更准确表述：**能算「项目定义的 utility proxy 在历史真实路径上的实现值」**，不是真实经济效用。

---

### 8.2 Worker「申请且获胜」比例，历史里能算吗？

**能**，且比 utility 更直接——纯统计，不依赖仿真。

#### 从历史日志直接算（推荐 baseline）

数据来源：`entries_by_worker` / `entries_by_project`（`src/dataset.py`）

| 指标 | 算法 |
|------|------|
| Worker `w` 申请次数 | 其非 withdrawn 的 entry 数 |
| Worker `w` 获胜次数 | `winner=True` 的 entry 数 |
| **逐人胜率** | `wins_w / applies_w` |
| **全局申请胜率** | `Σ wins / Σ applies` |
| 有申请的 worker 数 | 至少 1 次 entry 的 worker |
| **人均申请数** | `Σ applies / \|workers with applies\|` |

Requester 侧无「申请」概念，但有：每项目申请人数（entry 数）、是否产生 winner、项目级竞争度。

#### 仿真中的同类指标（公式相同、世界不同）

`env/platform_env.py` → `_worker_apply_win_metrics()` 统计的是**仿真策略下**的申请与中标（**非历史 hit**）：

- `aggregate_apply_win_rate` = 仿真总 wins / 仿真总 applies
- `mean_worker_apply_win_rate` / `median_worker_apply_win_rate` = 逐人胜率均值/中位数

与历史的**定义相同**，样本来自**仿真里的申请/选人流程**。

#### 与「历史 hit」的区别

| 概念 | 含义 |
|------|------|
| **历史 hit** | 策略选择是否等于日志 ground truth（后验复现） |
| **Apply→Win（仿真）** | 仿真中申请后被选为 winner 的比例（策略行为结果） |
| **Apply→Win（历史）** | 日志中 entry 最终 `winner=True` 的比例（真人当年路径） |

三者不可混用。

---

### 8.3 拿「项目结果」和「历史数据」对比，有意义吗？

**有意义，但要分清比什么、不能推出什么。**

#### 有意义的对比

**① 同口径 utility：历史路径 vs 仿真路径**

```
历史 baseline：对每条真实 (w,p,t) 算 U^W；对每个真实 winner 算 U^R → 取平均
仿真结果：    test eval 的 avg_worker_utility / avg_requester_utility
```

回答：在**同一 proxy** 下，仿真策略是否相对「真人当年实际选择」更优。

**② 申请胜率：历史 vs 仿真**

```
历史：从 entry 表算 global / per-worker apply-win rate
仿真：comparison.csv 中的 aggregate_apply_win_rate 等
```

回答：仿真是否系统性恶化或改善了 Worker **中标体验**（公平性 / 挫败感）。

**③ Hit vs 历史（项目已在用）**

回答：**行为复现度**，与 utility 优化正交。

#### 需谨慎、不宜过度解读

| 误区 | 原因 |
|------|------|
| 「仿真 utility > 历史 utility ⇒ 平台更好」 | 仿真 MDP ≠ 真实平台（batch、WAIT、recall、DQN 选人等） |
| 「仿真 apply-win 低 ⇒ 策略失败」 | 仿真若改变申请模式，胜率分母变，不可直接比绝对值 |
| 「历史 hit 高 ⇒ 真实福利高」 | Hit 是复现日志，非规范最优 |
| 把历史 utility 当 ground truth 最优 | 历史只是**一种**行为均衡，未必 maximize 你们的 U |

**根本限制**：日志只记录**实际发生的 (w,p)**；仿真改变了申请集合、等待、选人顺序——两个世界**样本不同**，只能比**同定义指标的相对水平**，不能作严格因果 A/B。

---

### 8.4 建议的三种 baseline（若做分析）

#### A. 历史 realized utility（描述性 baseline）

```text
对 test split 每条真实 entry：
  U_hist_worker += U^W(worker, project, entry_time)
对每个 test 项目的历史 winner：
  U_hist_requester += U^R(project, winner, t_select)
→ avg_worker_utility_hist, avg_requester_utility_hist
```

与 DQN / 启发式 test 的 `avg_*_utility` 同表对比。

#### B. 历史 apply-win（体验 baseline）

```text
global:  winners / entries
per-worker mean/median of (wins/applies)
```

与仿真 `aggregate_apply_win_rate`、`mean_worker_apply_win_rate` 并排。

#### C. Oracle / counterfactual（上限参考，需额外脚本）

在每个 worker 事件时刻 `t`：复现 recall 规则得到 active 候选，比较

- `U(真人选择)` vs `U(候选内 argmax)` vs `U(DQN 选择)`

分离 **proxy 上限**、**历史行为 gap**、**策略 gap**；工作量大但解释力最强。

---

### 8.5 结论摘要

| 问题 | 答案 |
|------|------|
| 历史能否算项目定义的 worker/requester 利益？ | **能**；用 `_compute_worker/requester_utility` 对真实 (w,p,t) 回放；与仿真同口径，但是 **proxy** |
| 能否算申请且获胜比例？ | **能**；历史从 entry/winner 直接统计；仿真从 `_worker_apply_win_metrics` 出 |
| 与项目结果对比有意义吗？ | **有**——同指标、不同机制下的参照；适合答「proxy 下是否更好」「体验是否更差」；**不能**当因果证明 |

**一句话**：历史记录足够构造 utility proxy 与 apply-win 的**离线 baseline**；与仿真对比有价值，但应表述为「固定 proxy / 固定指标定义下，仿真相对历史 baseline 的位置」，而非「真实平台优劣已证」。

**待实现**：`scripts/historical_baseline.py`（test split → utility + apply-win 表，与 `comparison.csv` 并排）。

---

## 9. 项目异质诉求：「等高质量」vs「快速选人」

### 9.1 问题

不同项目的 Requester 诉求可能不同：

- **高质量慢选型**：愿意较长等待，换取更高 worker quality、更大申请池；
- **快速够用型**：申请池有少量合格工人即可关闭，对 quality 要求相对低。

当前 MDP **未显式**建模这种异质性。

### 9.2 当前项目里实际有什么

#### 已有，但对所有项目「一刀切」

| 机制 | 当前做法 | 局限 |
|------|----------|------|
| 等待惩罚 | 全局 `project_wait_penalty=0.05` | 急单、慢单扣一样多 |
| 触发选人 | 全局 `requester_batch_size=8`、`requester_deadline_buffer_hours=24` | 所有项目同一套「攒够再选 / 临近 deadline 再选」 |
| Requester utility 权重 | 全局固定（quality 0.40 等） | 无法区分高质量诉求 vs 快速选人诉求 |
| WAIT 动作 | 非 deadline 强制时均可 WAIT | 无「本项目禁止久等」类约束 |

#### 已有，可间接反映差异（靠 DQN 隐式学习）

Requester **17 维 context 特征**（`_platform_project_context_features`）含：

- `hours_left`、`total_awards`、`featured`、`average_score`
- `wait_days`、申请池规模
- 申请池质量统计（`pool_mean_q`, `pool_max_q`, `pool_std_q`, `pool_top_gap`）

同一 Requester-DQN **理论上**可对这些项目多 WAIT、对另一些早选。  
Run B 中 test **`wait=24.4` 偏高**，说明隐式学习**不稳定**——无显式分类型设计时易收敛到「平均策略」。

#### 基线对照（策略级，非项目级）

启发式有 `wait_until_deadline` vs 立即 `worker_quality`，但是**整站统一策略**，不是「A 项目慢选、B 项目快选」。

### 9.3 业务上缺什么

**异质 Requester 偏好（heterogeneous requester preferences）**：

```
高质量慢选型：愿意等多天 → 更高 quality / 更大 pool
快速够用型：  池子 2–3 人尚可 → 早关，quality 权重低
```

当前所有 project 共用同一 Requester Agent、同一 reward 与 batch 规则 → **无法保证**按项目差异化。

### 9.4 可行处理方向（由易到难）

#### A. 项目级参数（最易落地）

为每个 project（或每类）配置：

```python
project_wait_penalty[p]       # 急单大、慢单小
requester_batch_size[p]       # 慢选型 16，快选型 3
requester_deadline_buffer_hours[p]
utility_quality_weight[p]     # 高质量型 0.6，快速型 0.2
```

**类型来源**：规则（`featured` / 高奖金 → 慢选；短 deadline → 快选）、按 category 聚类、或客户标注 tier。

#### B. 把「项目诉求」写进 state（条件策略）

在 Requester context 增加：

- `project_tier` one-hot（快 / 慢 / 标准）
- `quality_vs_speed_preference`
- `deadline_pressure = 1 / hours_left`

学 **π(select | pool, project_type)**，比仅依赖 `featured`、`hours_left` 更明确。

#### C. 项目异质化的 utility / reward

\[
U^R_p = \alpha_p \cdot Q_{worker} + \beta_p \cdot \mathbb{E}[score] + \cdots - \lambda_p \cdot \text{wait}
\]

- **高质量型**：\(\alpha_p\) 大、\(\lambda_p\) 小 → 鼓励 WAIT、挑更好的人  
- **快速型**：\(\lambda_p\) 大、池子 ≥ k 即倾向早关  

WAIT 的 Q 值随 \(\lambda_p\) 变化：急单 WAIT 更亏，慢单 WAIT 相对可接受。

#### D. 双策略 / 分层 Requester

- **上层**：规则或分类器 → project 映射到 `{quality_first, speed_first}`  
- **下层**：两套 Requester 策略，或同一网络 + type embedding  

「先分诊、再选人」，工程上常比单一 DQN 稳。

#### E. Contextual bandit / 多任务 RL

以 `(category, award, deadline, featured)` 为 context，学 optimal batch / wait / quality 阈值。适合类型多、规则难写全的场景。

#### F. 机制设计（不只靠 RL）

平台规则层硬约束，RL 只在窗口内排序 worker：

- **早关触发**：池子达「最低可用质量」且人数 ≥ k → 自动选人（快单）  
- **最低等待期**：高质量项目强制 WAIT 至 batch≥N（慢单）  
- **动态 batch**：`batch_size = f(award, hours_left, featured)`  

往往比纯学 WAIT 更可靠。

### 9.5 与 Run B 现象的关系

Run B 中 Requester utility 高但 **wait 高、5 个 unfilled**，部分因为：

- 全局 wait 惩罚偏弱，WAIT 相对 utility 收益「太便宜」  
- 全局 `batch=8`，不区分小急单与大慢单  
- 单一 Agent 在异质项目上学平均策略 → 部分 over-wait、部分 under-fill  

引入 **项目异质化 \(\lambda_p\)、\(batch_p\)** 有机会同时改善 wait 与 fill，而不必牺牲高质量项目的选人空间。

### 9.6 建议落地顺序

| 步骤 | 内容 |
|------|------|
| 1 | 规则分 tier（featured / 奖金 / deadline 压力）→ 三套 `wait_penalty`、`batch_size` |
| 2 | Context 加 `project_tier` one-hot → 同一 Requester-DQN 学条件策略 |
| 3 | **按 tier 拆开评估**：慢选型的 wait、winner_quality；快选型的 time-to-fill、fill_rate |
| 4 | 仍不稳则加 **机制层硬约束**（方案 F） |

### 9.7 结论摘要

| 问题 | 答案 |
|------|------|
| 当前是否处理了「有的项目愿等、有的要快」？ | **没有显式处理**；仅全局参数 + context 特征隐含差异 |
| Requester 能否自己学出来？ | **可能**，但 Run B 表明学得不稳 |
| 推荐路径 | 项目级 \(\lambda_p、batch_p、utility 权重\) → context 条件策略 → 机制约束 |

**一句话**：把「质量–速度权衡」从**全局常数**改成**随 project 变化的参数或 context**，是改善 wait/fill 错位、贴合真实平台诉求的关键方向之一。

---

## 10. Reward 设计问题（Run B / `reward_mode=utility`）

讨论对象：Worker / Requester **Agent 逐步收到的训练 reward**（非 `platform_reward` 累计指标）。

### 10.1 核心矛盾：优化 proxy，不是平台运营结果

| 训练优化 | 平台更关心 |
|----------|------------|
| `U_worker` / `U_requester`（手工公式） | wait、fill、双边体验、流程稳定 |
| 决策瞬间可算的 proxy | 项目是否关闭、worker 是否白等 |

Run B：**test utility 合计最高（0.977），platform 低于 industry_match，wait 最高，5 unfilled** → reward 在优化与运营目标**部分重合、但不等价**的 surrogate。

### 10.2 Worker reward：申请即给分，落选无反馈

Worker 在**提交申请当下**获得 reward（`utility` + 可选 `legacy_hit_weight=0.1`），项目关闭、未中标 reroute 时**无负 reward**。

后果：

- 鼓励投「proxy 高」项目，不管能否中标 → **recall@k 高、hit 低**。
- 多次落选的伤害**进不了训练信号**（见 §5）。

### 10.3 Requester reward：WAIT 与 utility 权衡失衡

Requester Agent reward = `U_requester - wait_cost`（全局 `project_wait_penalty=0.05/天`）。

问题：

1. Wait 惩罚相对 utility 偏弱 → WAIT「太便宜」→ test **wait=24.4**。
2. 全局单一系数，无项目异质（§9）。
3. **Unfilled**（`_close_unfilled`）几乎**不给 Requester Agent 明确惩罚**，fill 与 RL 信号错位。

### 10.4 双边各自为政

- Worker-DQN 最大化 `U^W`；Requester-DQN 最大化 `U^R - wait`。
- 无联合 social welfare → **双边各自 utility 高 ≠ 平台整体好**。

### 10.5 Utility 公式与 hit bonus

- Utility 权重（0.35/0.25/…）**经验设定**，未用 platform/wait/fill 校准。
- Utility 基本不绑真实 outcome；与历史 hit 可严重偏离。
- Utility 模式下 `hit` 仍来自 `outcome_for`（+0.1），混入了**后验标签**，概念上不纯（量级小）。

### 10.6 信用分配（credit assignment）

| 事件 | 何时给 reward | 问题 |
|------|---------------|------|
| Worker 申请 | 立即 | 与最终结果脱节 |
| Requester WAIT | 扣 wait | 偏弱 |
| Requester 选人 | 立即 utility | 不依赖后续运营结果 |
| Worker 落选 reroute | **无** | 学不到「这次亏了」 |
| 项目 unfilled | **基本无** Agent 罚 | 与 fill 指标错位 |

### 10.7 与 checkpoint 的不一致

`validation_score` 主看 utility + 弱 hit/recall，**几乎不看** wait、fill、platform → 易选 **utility 高但 wait 高** 的 ep10。

### 10.8 改进方向（摘要）

| 优先级 | 方向 |
|--------|------|
| 高 | Worker 落选 delayed penalty；utility 加 streak / 期望胜率 |
| 高 | 加大 wait / unfilled 惩罚；项目 tier 异质 \(\lambda_p\) |
| 中 | Checkpoint 加入 wait、fill、fairness |
| 中 | BC / 标签改为校准期望收益 |

---

## 11. `platform_reward` 设计问题

`platform_reward` 是 **episode 内逐步累加的评估标量**，不是单独设计的平台目标函数，而是 Agent 步 reward 的副产品。

### 11.1 定义（代码口径）

**Worker 步**：

```text
platform_reward += worker_reward    # ≈ U_worker (+ 0.1·hit)
```

**Requester 步**：

```text
platform_reward += requester_reward - wait_cost    # ≈ U_requester - wait
```

形式化：

\[
\text{platform\_reward} = \sum_{\text{worker 步}} r^W + \sum_{\text{requester 步}} (r^R - \text{wait\_cost})
\]

**注意**：`final_metrics` 中分项 `worker_reward`、`requester_reward` 为**原始累计**；`requester_reward` **不含** wait 扣减。因此：

\[
\text{platform\_reward} \neq \text{worker\_reward} + \text{requester\_reward} - \text{project\_wait\_cost}
\]

报告若写「platform = worker + requester − wait」过于简化。

### 11.2 非训练目标，却常作 headline KPI

- DQN **不直接优化** `platform_reward`；Run B checkpoint 按 utility 综合分选取。
- 基线对比、报告结论却常用 platform 排序 → **优化目标与 headline 指标分裂**。

### 11.3 累计量失真：步数与回流「刷分」

- **Episode 全程求和**，非 per-project / per-step 效率（虽有 `platform_reward_per_project`、`platform_reward_per_step`，报告少用）。
- Worker 步 **远多于** Requester 步 → 累计值主要由 Worker **反复申请** utility 堆高。
- 落选 worker **reroute 后再申请** → 同一人多次拿申请 reward；**回流越多，platform 可能越高**。
- 策略 **总步数不同**（Run B：DQN 3026 vs 启发式 3312）→ 累计值**不可严格比效率**。

**典型案例**：`random_project + wait_until_deadline` 的 platform **~913** 远高于正常策略，伴随 `rerouted_workers=4517`、30 unfilled、极高 wait —— **空转刷分**，非业务最优。

### 11.4 与 `project_wait_cost` / unfilled 口径不一致

| 场景 | `project_wait_cost` | `platform_reward` |
|------|-------------------|-------------------|
| Requester WAIT / 选人 | 累加 wait | 该步扣 wait |
| **Unfilled 关项目**（`_close_unfilled`） | 累加 wait | **不更新** |

→ unfilled 的等待代价在 wait 指标里可见，在 platform 累计里**几乎不可见**；Run B 的 **5 unfilled** 会使 platform 排名**相对偏乐观**。

### 11.5 组成偏斜：Worker「申请即加分」主导

Worker 申请即 utility、落选不扣；Requester 才有 wait 惩罚。  
`platform_reward` 更像「申请 proxy 之和 + 选人 utility − 部分 wait」，**不是**平衡的平台运营得分。

Run B：requester 分项可很高，platform 仍因 worker 偏低 + wait 偏高而落后启发式。

### 11.6 缺少平台硬约束项

`platform_reward` **未显式包含**（或极弱）：fill/unfilled 惩罚、双边公平、落选成本、项目异质、winner 实际质量（仅间接 via utility）。

完整 wait 在 **`project_wait_cost`**，fill 在 **`filled_project_rate`** —— 必须**分列**，单看 platform 会误判。

### 11.7 与 avg utility 的关系混乱

| 指标 | 含义 |
|------|------|
| `avg_worker/requester_utility` | 决策步**平均** proxy |
| `platform_reward` | 全程**求和**，含步数、回流、wait |

尺度不同（train platform ~2800–3350 vs avg utility ~0.3–0.6）；**不是**简单单调关系。

### 11.8 评估用法上的问题

1. 跨策略比**累计** platform：步数、fill 路径不同时易误导。
2. **Train platform 下降**当失败：train filled ~79%、路径更长，不全是策略变差。
3. Run A 用 val platform 选模、Run B 用 utility 选模 → 同一指标角色不一致。
4. 应主表并列：`platform_reward`、`platform_reward_per_project`、`project_wait_cost`、`filled_rate`、`steps`、`rerouted_workers`。

### 11.9 问题汇总

| # | Reward / platform 共有或分列问题 | Run B 表现 |
|---|----------------------------------|------------|
| 1 | Proxy 与 platform 错位 | utility↑ platform↓ |
| 2 | Worker 申请即奖、落选无罚 | hit 低 |
| 3 | Wait 弱、无项目异质 | wait=24.4 |
| 4 | Unfilled 弱 RL / platform 信号 | 5 unfilled |
| 5 | 双边独立优化 | req_U 高 + wait/fill 差 |
| 6 | platform 累计 + 回流刷分 | random-wait 异常高 |
| 7 | unfilled wait 不进 platform | 与 wait 指标不一致 |
| 8 | platform ≠ worker_req − wait | 分项口径混用 |
| 9 | Checkpoint 不盯 wait/fill | ep10 utility 峰 |

### 11.10 若保留 `platform_reward` 的改进建议

1. **主报归一化**：`platform_reward_per_project` / `per_step` + 附 `steps`。
2. **补项**：unfilled 惩罚、落选惩罚；与 `project_wait_cost` 口径对齐。
3. **防刷分**：重复申请递减；或改用均值/折扣而非裸累加。
4. **训练对齐（可选）**：checkpoint 参考 platform 分解项。
5. **报告规范**：platform 与 utility、wait、fill **分列**，不单列 platform 论优劣。

### 11.11 结论摘要

| 概念 | 一句话 |
|------|--------|
| **Agent reward** | 逐步 utility（± wait / 弱 hit）；训练优化对象；存在申请即奖、落选无罚、wait 弱等问题 |
| **platform_reward** | Agent 步 reward 的 **episode 累计**；**非训练目标**；有累加失真、回流刷分、unfilled 未入账等问题 |
| **怎么用** | utility 看 proxy 是否学到；platform + wait + fill + steps 看运营；二者分列，不混为一个「好不好」 |

---

## 13. 待进一步考虑的方向

> 主线（proxy、reward、platform、wait/fill、异质项目、落选体验）已有多节覆盖。以下为**尚未系统展开**、但对结论可信度与后续迭代仍有价值的话题。

### 13.1 与结论可信度直接相关（建议优先）

#### 离线策略评估（OPE）与反事实

- Test 是在**改过的仿真机制**里 rollout，不是真实平台 A/B。
- 日志里**未出现的 (worker, project)** 占绝大多数 → reward / utility 多为外推。
- 可考虑：IPS / Doubly Robust 等 OPE；或对关键策略 **bootstrap / 多 seed** 看排名是否稳定。
- 报告应明确：**test 是 simulation benchmark，不是真实因果效应**。

#### 双 Agent 非平稳性

- Worker 与 Requester **同时学习**，各自转移随对方策略变化；独立 Q-learning **无 general-sum 收敛保证**。
- ep10 后 utility 回落可能含**非平稳震荡**，不单是 underfitting。
- 可试：**freeze 一侧**（固定启发式 Requester + 只训 Worker）、**交替训练**、分阶段训练。
- 理论局限宜写进报告「未来工作 / 局限」。

#### 探索与 BC 的交互

- Worker ε ep1 触底、BC 后几乎不 explore → **结构问题**，非仅调参。
- BC 分布 ≠ RL 在线分布 → **分布偏移**。
- 开放：BC 后是否重置 ε？是否只 BC Requester？双边探索节奏是否应匹配？

#### Recall 与 Ranking 的归因

- 已有 recall@k 高、hit 低，但未量化：**多少损失来自 mixed recall**，多少来自 DQN 排序。
- 建议：**Oracle 上界**（候选内按 utility 或 truth 排序）作 ceiling。
- Worker（全局 mixed recall）与 Requester（申请池召回）**不对称**带来的系统偏差值得单独分析。

---

### 13.2 机制与仿真有效性

#### 仿真 vs 真实 Crowdspring

| 仿真假设 | 待核对 |
|----------|--------|
| Worker `busy` 期间只能挂一个 project | 真实是否允许多项目并行关注/申请？ |
| Requester batch=8、WAIT | 与真实选人/UI/通知机制是否可比？ |
| `synthetic` reroute 事件 | 日志无直接对应；回流频率是否失真？ |

报告宜加 **「仿真简化假设」** 小节，避免读者默认等于真实平台复刻。

#### 冷启动与长尾

- 新 Worker（`past_count=0`）skill 退化公式是否合理？
- 零申请池、新 Project 上 Requester 行为？
- 长尾 category / 小样本 industry 上 utility 与 hit 是否更差？

#### 时间漂移与外推

- train/val/test 按 `start_date` 切分已做；test 期分布是否**漂移**（类目、奖金、竞争）？
- 策略是否仅适用于训练期平台生态？

---

### 13.3 目标与算法（若继续迭代）

#### 中心化 / 协调式平台目标

- **Stackelberg**、**central planner** 最大化 `α·U_w + β·U_r − wait − unfilled`
- **约束 RL**：fill_rate ≥ 阈值、wait ≤ 上限
- 比单纯调 MLP / lr 更有方向感；课程项目可写「未来工作」。

#### 算法替代与 baseline 完整性

- Contextual bandit（稀疏 winner 是否更合适？）
- Worker 侧 **LTR / 监督排序** 强 baseline
- Legacy hit vs utility **系统对照**是否跑全
- 离线 RL（如 conservative Q）缓解 OOD 动作过高 Q

#### 机制层默认策略

- 默认 batch / buffer 是否应 **规则先行**，RL 只做池内排序？
- 部分 project 对 WAIT **禁用或封顶**（硬约束常比学出来更稳）→ 与 §9 衔接。

---

### 13.4 公平、体验与产品（扩展 §5）

- **中标集中度**：少数高质量 Worker 是否垄断 winner？
- **Gini / 人均 apply-win 分布**（历史 vs 仿真）
- Requester fairness 若加入 utility，如何防刷低质量项目凑 win？
- Worker **并行申请**若允许，对 reward 与 platform 含义的影响
- 等待期「无反馈」的产品层解法（仿真外）

---

### 13.5 工程、复现与报告规范

#### 指标主表规范（制度化 §11）

建议固定最少列：  
`avg_U_w, avg_U_r, platform_per_project, wait, fill, steps, rerouted, worker_hit, apply-win`  
避免各章各挑一列得出矛盾结论。

#### 消融与 sensitivity（尚未系统做）

- 去掉 mixed recall / lookahead / batch requester 各一项
- utility 权重 ±20%；`wait_penalty` ×2、×5
- 有无 BC  
→ 2–3 个 ablation 往往比再加 10 epoch 更有说服力。

#### 复现与随机性

- 环境 seed、训练 seed；**排名对 seed 是否敏感**
- ep10 utility 峰是否随 seed 稳定

#### 伦理与数据使用（可选）

- 历史 winner 作 hit / outcome 的事后标签说明
- 若部署：utility 权重谁定、如何审计

---

### 13.6 建议的下一步优先级

| 优先级 | 话题 | 理由 |
|--------|------|------|
| ★★★ | 仿真假设 vs 真平台 | 界定结论适用范围 |
| ★★★ | Recall vs 排序 + oracle 上界 | 知改进该改召回还是 RL |
| ★★★ | 双 Agent 非平稳 + BC/ε | 解释 train 曲线与 ep10 |
| ★★ | 历史 utility / apply-win baseline | 纪要已有，差脚本与数字 |
| ★★ | 指标主表 + per_project 归一化 | 避免 platform 误导 |
| ★★ | 2–3 个关键 ablation | 支撑设计选择 |
| ★ | OPE / 多 seed 稳健性 | 提升学术严谨度 |
| ★ | 公平性 / 冷启动 / 时间漂移 | 扩展讨论深度 |

**一句话**：除已讨论主线外，最宜优先补 **仿真边界、recall/排序归因、双 Agent 非平稳、历史 baseline 数字、消融与指标规范**——以减少「实验很多但不知信哪条结论」的风险。

---

## 14. 参考代码位置

| 主题 | 文件 |
|------|------|
| Utility / Hit / Legacy reward | `env/platform_env.py` |
| Worker 候选 mixed recall | `env/platform_env.py` → `_build_project_candidates`, `_mixed_project_recall` |
| Requester 工人池召回 | `env/platform_env.py` → `_build_requester_worker_pool` |
| Q 网络结构 | `models/dqn.py` |
| 训练超参 / validation_score | `scripts/train_platform_dqn.py` |
| BC 预训练 | `scripts/pretrained_platform_bc.py` |
| 特征编码 | `src/features.py` |
| 历史 entry / worker 事件 | `src/dataset.py` → `EntryRecord`, `iter_worker_events` |
| 历史 outcome 表 | `src/platform_dataset.py` → `outcome_for`, `_build_outcomes` |
| Requester 触发 / WAIT / batch | `env/platform_env.py` → `_should_trigger_requester`, `_step_requester` |
| Requester 17 维 context | `env/platform_env.py` → `_platform_project_context_features` |
| 全局 wait / batch 配置 | `env/platform_env.py` → `PlatformEnvConfig` |
| Requester 启发式（含 wait_until_deadline） | `models/platform_baselines.py` |
| Worker / Requester 步 reward 与 platform 累计 | `env/platform_env.py` → `_step_worker`, `_step_requester`, `final_metrics` |
| Unfilled 关项目 | `env/platform_env.py` → `_close_unfilled` |
| Wait 成本 | `env/platform_env.py` → `_apply_wait_cost` |
| 归一化 platform 指标 | `env/platform_env.py` → `platform_reward_per_project`, `platform_reward_per_step` |
| 正式实验报告 | `platform_dqn_experiment_report_run_b.md` |

---

*文档版本：2026-05-30（§10–11 Reward/platform_reward；§13 待考虑方向增补）。*
