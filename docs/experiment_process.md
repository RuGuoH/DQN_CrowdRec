# 模型实验过程记录

> 用途：只记录与模型设计、训练效果、评估结果相关的困难、调整目的、具体调整、指标变化与可能原因。该文件服务于最终实验报告和分组汇报，不替代 `runs/` 中的原始指标文件。

## 记录原则

- 每次训练、评估、消融，或会影响模型结果解释的调整，都追加一条记录。
- 记录范围限于模型相关问题，例如状态/动作/奖励设计、候选集构造、特征工程、Q 网络结构、DQN 变体、超参数、训练不稳定、指标异常、基线对比异常等。
- 不记录工程性问题，例如 conda/PyTorch 环境、依赖安装、路径、缓存、日志目录、文件是否写入项目等，除非它直接改变了模型输入、训练设置或评价指标。
- 不编造实验数字；只记录实际运行命令、输出摘要、日志路径或明确写“待验证”。
- 重点写清“为什么调整”，而不是只写“改了什么”。
- 若调整没有改善，也要记录，因为失败过程可以解释最终设计选择。

## 记录模板

```markdown
### YYYY-MM-DD 实验名称

- 目的：
- 数据与设置：
- 命令：
- 模型相关困难/现象：
- 调整目的：
- 具体调整：
- 指标变化：
- 可能原因：
- 针对本次尝试的改进方向：
```

## 当前记录

### 2026-05-28 动态双边平台环境重构

- 目的：回应作业中“参与者和发布者是动态变化”的要求，把原先相互独立的 worker 侧任务推荐和 requester 侧 worker 推荐，改为同一个动态平台仿真实验。
- 数据与设置：保留原始 Crowdspring 数据读取；新增统一 `PlatformDataset`，按 worker 到达时间推进；project 在 deadline 前可累计申请 worker；Requester-DQN 每次新申请到达后可选择 `WAIT` 或选 winner；未中标 worker 生成 synthetic release event 并重新进入推荐队列。
- 命令：
  - 启发式 smoke：`python scripts/evaluate_platform.py --split train --max-projects 50 --worker-policy random_project --requester-policy select_first --max-steps 20 --output runs\tmp_platform_smoke_eval.json`
  - 批量基线 smoke：`python scripts/run_platform_baselines.py --split train --max-projects 50 --max-steps 20 --output-dir runs\tmp_platform_baselines`
- 模型相关困难/现象：
  - 原有实验只是在两个独立 MDP 中做候选集内排序，不能表达 project 申请池、winner 决策、未中标 worker 回流和 project 等待成本。
  - 原有 requester 侧环境每一步直接预测历史投稿 worker，没有“继续等待”的动作，也没有 project 关闭状态。
  - 默认强制注入 truth 的候选集设置更适合诊断排序上限，不适合作为动态平台主结果。
- 调整目的：把“参与者收益”和“发布者收益”放入同一个平台状态转移中，同时保留两个 DQN 分别学习 worker 推荐和 requester 选人/等待决策。
- 具体调整：
  - 新增 `src/platform_dataset.py`，统一 worker 到达事件、project 状态和 worker-project outcome。
  - 新增 `env/platform_env.py`，实现 worker 到达、project 申请池、Requester-DQN 的 `WAIT`/选人动作、deadline 强制决策、project 关闭和未中标 worker 回流。
  - 新增 `models/platform_training.py`，支持两个 DQN 的异步 transition：每个 agent 的 transition 只连接到该 agent 的下一次决策。
  - 新增 `models/platform_baselines.py` 与三个主入口脚本：`train_platform_dqn.py`、`evaluate_platform.py`、`run_platform_baselines.py`。
  - README 与报告大纲改为以 platform 联合仿真为主线，旧 worker/requester 实验保留为 legacy 对照。
- 指标变化：
  - 启发式单次 smoke 在 50 项目、20 步上可输出 `platform_reward=8.41`、`worker_hit_rate=0.5000`、`requester_hit_rate=0.0000`。
  - 批量基线 smoke 能输出 `comparison.csv/json`；在 20 步调试口径下，`random_project+wait_until_deadline` 的 `platform_reward=10.37`，其余多数组合约为 `8.41`。
  - 回流机制 smoke 使用 `wait_until_deadline` 跑 200 步时，`rerouted_workers=25`，说明 project 关闭后未中标 worker 已能重新进入推荐队列。
  - DQN 训练入口已编译通过，但当前默认 Python 环境缺少 PyTorch，未在本轮执行训练 smoke。
- 可能原因：
  - 小步数 smoke 中 requester hit 为 0，主要因为 winner 信号稀疏，且 20 步内多数 project 仍处于申请池或等待/早选状态。
  - `wait_until_deadline` 在短步数中 platform reward 较高，可能是因为等待策略减少了过早选择非历史 winner 的惩罚；正式结论需在全量 test 和足够步数下确认。
  - 新环境默认不注入 truth，worker hit 会显著低于旧候选集排序实验，但更接近真实动态推荐。
- 针对本次尝试的改进方向：
  - 在有 PyTorch 的环境中运行 `python scripts/train_platform_dqn.py --max-projects 50 --episodes 1 --max-steps 100`，确认两个 DQN checkpoint 与 metrics 正常生成。
  - 全量实验优先比较 `Platform DQN`、`JointHeuristic` 和 `LowWait-Quality`，重点观察 `platform_reward`、`project_wait_cost`、`filled_project_rate` 和 `rerouted_workers`。
  - 若 requester hit 长期接近 0，应调高 winner/finalist reward 或加入监督式预训练，让 Requester-DQN 先学会从申请池中识别历史 winner。

### 2026-05-28 Platform Dueling + Double DQN 全量训练与作图

- 目的：在动态双边平台环境上训练第一版双 agent DQN 模型，并输出训练过程数据、学习曲线和 test 集对比结果。
- 数据与设置：全量数据 `max_projects=0`；动态平台环境；`include_truth_in_candidates=False`；Worker-DQN 与 Requester-DQN 均使用 `model_type=dueling` 和 `double_dqn=True`；`device=cuda`；`episodes=10`；每轮最多 800 步。
- 命令：
  - 训练：`C:\Users\17765\.conda\envs\torch\python.exe -u scripts\train_platform_dqn.py --max-projects 0 --episodes 10 --device cuda --worker-model dueling --requester-model dueling --worker-double-dqn --requester-double-dqn`
  - 作图：`C:\Users\17765\.conda\envs\torch\python.exe scripts\plot_platform_training.py runs\platform\platform_dqn_no_truth_20260528_212013\metrics.csv`
  - test 评估：`C:\Users\17765\.conda\envs\torch\python.exe scripts\evaluate_platform.py --split test --max-projects 0 --worker-policy dqn --requester-policy dqn --worker-checkpoint runs\platform\platform_dqn_no_truth_20260528_212013\checkpoints\worker_best.pt --requester-checkpoint runs\platform\platform_dqn_no_truth_20260528_212013\checkpoints\requester_best.pt --output runs\platform\platform_dqn_no_truth_20260528_212013\test_eval_best.json`
- 模型相关困难/现象：
  - 训练阶段 worker hit_rate 在 0.05 到 0.08 之间波动，验证 worker hit_rate 最高约 0.0627，说明未注入 truth 的动态候选集下任务推荐更难。
  - requester hit_rate 长期很低，验证集约 0.0027，test 集约 0.0109，说明从申请池中识别历史 winner 的信号仍然稀疏。
  - 验证集 platform_reward 在第 2 个 episode 达到最高，后续没有持续提升，当前 10 个 episode 更像初始可运行模型，而非充分收敛模型。
- 调整目的：使用更稳的 Dueling + Double DQN 作为动态平台主线初始模型，并记录完整训练曲线以便后续调参与报告绘图。
- 具体调整：
  - 新增 `scripts/plot_platform_training.py`，从 `metrics.csv` 输出 reward、hit_rate、动态状态、loss、epsilon 曲线和 `training_summary.csv`。
  - 使用 `runs\platform\platform_dqn_no_truth_20260528_212013\checkpoints\worker_best.pt` 与 `requester_best.pt` 做 test 评估。
  - 生成含 DQN 的基线对比表：`runs\platform_baselines_with_dqn\platform_test_no_truth\comparison.csv`。
- 指标变化：
  - 训练集最佳 platform_reward：episode 4，`platform_reward=144.9044`，`worker_hit_rate=0.0750`，`requester_hit_rate=0.0050`。
  - 验证集最佳 platform_reward：episode 2，`platform_reward=173.8011`，`worker_hit_rate=0.0627`，`requester_hit_rate=0.0027`。
  - 最终 episode 10：训练集 `platform_reward=140.7379`，验证集 `platform_reward=171.6948`。
  - test 集 DQN：`worker_hit_rate=0.09239`，`requester_hit_rate=0.01087`，`platform_reward=197.7825`，`project_wait_cost=0.8193`，`filled_project_rate=1.0`。
  - test 对比中 DQN 的 platform_reward 高于 `industry_match+worker_industry_match` 的 195.0816、`low_wait_project+worker_quality` 的 194.2817、`joint_heuristic+worker_quality` 的 192.3024，但低于 `random_project+wait_until_deadline` 的 879.8873；后者伴随 `rerouted_workers=4517` 和较高等待成本，不能只按 reward 单指标解释。
- 可能原因：
  - Worker-DQN 在无 truth 注入时仍能略高于多数启发式 worker hit，说明模型学到了一部分候选 project 排序信号。
  - Requester-DQN 的 winner 识别较弱，可能是 winner/finalist 样本稀疏、每个 project 申请池较小、当前 reward 对 winner 区分度不足。
  - `random_project+wait_until_deadline` reward 异常高，可能来自等待到 deadline 后产生大量 worker 回流和更多 worker reward 累积；报告中应把它作为动态机制的异常基线，而不是简单视为最优策略。
- 针对本次尝试的改进方向：
  - 增加训练 episode 或缩短 epsilon 衰减，让验证集不只停留在高探索早期策略。
  - 给 Requester-DQN 增加监督式 winner 预训练或提高 winner/finalist 权重，改善 requester hit_rate。
  - 调整 platform_reward 中 worker 回流累计收益和 project 等待成本的权重，避免极端等待策略因回流次数多而获得过高总 reward。

### 2026-05-28 Platform reward 调整与上升趋势训练

- 目的：修正 requester reward 中“非 winner 被扣 `-0.1`”的不合理设定，并通过更快的 epsilon 衰减让训练曲线更明显体现 reward 上升趋势。
- 数据与设置：全量数据 `max_projects=0`；动态平台环境；`include_truth_in_candidates=False`；Worker-DQN 与 Requester-DQN 均使用 `Dueling + Double DQN`；`episodes=20`；每轮最多 800 步；`epsilon_decay_steps=1000`；`epsilon_end=0.05`。
- 命令：
  - 训练：`C:\Users\17765\.conda\envs\torch\python.exe -u scripts\train_platform_dqn.py --max-projects 0 --episodes 20 --device cuda --worker-model dueling --requester-model dueling --worker-double-dqn --requester-double-dqn --epsilon-decay-steps 1000 --log-dir runs\platform_adjusted`
  - 作图：`C:\Users\17765\.conda\envs\torch\python.exe scripts\plot_platform_training.py runs\platform_adjusted\platform_dqn_no_truth_20260528_220322\metrics.csv`
  - test 评估：`C:\Users\17765\.conda\envs\torch\python.exe scripts\evaluate_platform.py --split test --max-projects 0 --worker-policy dqn --requester-policy dqn --worker-checkpoint runs\platform_adjusted\platform_dqn_no_truth_20260528_220322\checkpoints\worker_best.pt --requester-checkpoint runs\platform_adjusted\platform_dqn_no_truth_20260528_220322\checkpoints\requester_best.pt --output runs\platform_adjusted\platform_dqn_no_truth_20260528_220322\test_eval_best.json`
- 模型相关困难/现象：
  - 原 reward 会把 requester 选到非历史 winner 的 worker 直接扣 `-0.1`，这会把“质量尚可但不是最终 winner”的选择和无效选择混在一起。
  - 默认 epsilon 衰减过慢时，20 轮以内训练曲线会长期受随机探索影响，不容易展示模型从探索到利用后的 reward 改善。
- 调整目的：
  - 让 requester reward 更符合“获得高质量投稿”的目标：非 winner 不再额外扣分，但仍由 worker quality、历史 score、winner/finalist、类目/行业匹配贡献正向信号。
  - 让训练后期更快进入低探索阶段，使 `platform_reward` 的上升趋势更明显。
- 具体调整：
  - 在 `env/platform_env.py` 中取消 requester 选择非 winner 时的 `miss_penalty=-0.1`；仅 winner 获得 `hit_reward + winner_bonus`，finalist 获得 `finalist_bonus`，普通非 winner 不额外加减 winner 惩罚。
  - 在 `scripts/train_platform_dqn.py` 新增 `--lr`、`--batch-size`、`--buffer-size`、`--target-update-freq`、`--epsilon-decay-steps`、`--epsilon-end`，并把默认 `epsilon_decay_steps` 调为 1000。
  - 在 `scripts/plot_platform_training.py` 新增 `reward_trend_curves.png`，同时绘制 `platform_reward` 的 running best 和 3-episode moving average，用于展示可达到 reward 的上升趋势。
- 指标变化：
  - 小规模调参 smoke：`max_projects=50` 时，train `platform_reward` 从 episode 1 的 `35.6953` 上升到 episode 5 的 `43.3346`，说明方向有效；但验证集项目太少，趋势不稳定。
  - 全量正式训练：train `platform_reward` 从 episode 1 的 `169.1802` 上升到 episode 20 的 `207.6611`，最高 episode 19 为 `211.6412`。
  - 全量正式训练的验证集最佳：episode 6，`val_platform_reward=213.2605`，`val_worker_hit_rate=0.0654`，`val_requester_hit_rate=0.00545`。
  - 全量正式训练最终：episode 20，`val_platform_reward=208.3248`，`val_worker_hit_rate=0.0596`，`val_requester_hit_rate=0.00271`。
  - test 集 best checkpoint：`platform_reward=229.4431`，`worker_hit_rate=0.08424`，`requester_hit_rate=0.01087`，`filled_project_rate=1.0`，`project_wait_cost=0.8193`。
  - 新 reward 口径下的 test 基线对比保存于 `runs\platform_baselines_adjusted\platform_test_no_truth\comparison.csv`：`dqn+dqn` 的 `platform_reward=229.4431`，低于 `industry_match+worker_industry_match` 的 `231.4816` 和 `low_wait_project+worker_quality` 的 `230.6817`；`random_project+wait_until_deadline` 仍异常较高，为 `913.5873`，但伴随 `rerouted_workers=4517`、`unfilled_projects=30` 和高等待成本。
  - 训练曲线和汇总表保存于 `runs\platform_adjusted\platform_dqn_no_truth_20260528_220322\plots\`，其中 `reward_trend_curves.png` 最直接体现 reward 上升。
- 可能原因：
  - 取消非 winner 惩罚后，Requester-DQN 不会因为选择一个高质量但非最终 winner 的 worker 而被过度惩罚，platform reward 的 requester 部分明显抬升。
  - `epsilon_decay_steps=1000` 使 episode 10 左右已经接近 `epsilon_end=0.05`，训练后半段更多体现当前 Q 网络的贪心策略，因此 train reward 的 running best 呈上升趋势。
  - 验证集 reward 没有单调上升，说明模型仍存在策略波动；报告中更适合同时展示原始曲线、moving average 和 running best，而不是声称每个 episode 都单调提升。
- 针对本次尝试的改进方向：
  - 下一轮可继续围绕 requester 做奖励消融：分别提高 winner bonus、score weight 或 worker quality weight，比较 requester hit 与 winner quality 是否同步提高。
  - 对 platform reward 做归一化或按 project 计均值，减少“关闭 project 数量”和“回流 worker 数量”对总 reward 的尺度影响。
  - 若报告需要更平滑的验证曲线，可以固定评估事件子集或增加 episode 数，让验证指标不被少量 winner 命中波动主导。

### 2026-05-29 Platform DQN 完整 episode 全量实验与报告数据

- 目的：按报告正式口径重新从头生成实验数据，不做小数据或步数截断 smoke；补齐数据统计、训练曲线、test 评估和基线对比。
- 数据与设置：全量数据 `max_projects=0`；动态平台环境；`include_truth_in_candidates=False`；Worker-DQN 与 Requester-DQN 均使用 `Dueling + Double DQN`；`episodes=20`；`max_steps=0`，即每个 episode 完整推进事件直到 train split 内 project 关闭；`epsilon_decay_steps=1000`。
- 命令：
  - 数据加载：`C:\Users\17765\.conda\envs\torch\python.exe -m src.dataset --max-projects 0`
  - 训练：`C:\Users\17765\.conda\envs\torch\python.exe -u scripts\train_platform_dqn.py --max-projects 0 --episodes 20 --max-steps 0 --device cuda --worker-model dueling --requester-model dueling --worker-double-dqn --requester-double-dqn --epsilon-decay-steps 1000 --log-dir runs\report_full_20260529\platform_dqn_full`
  - 作图：`C:\Users\17765\.conda\envs\torch\python.exe scripts\plot_platform_training.py runs\report_full_20260529\platform_dqn_full\platform_dqn_no_truth_20260529_205032\metrics.csv`
  - test 评估：`C:\Users\17765\.conda\envs\torch\python.exe scripts\evaluate_platform.py --split test --max-projects 0 --worker-policy dqn --requester-policy dqn --worker-checkpoint runs\report_full_20260529\platform_dqn_full\platform_dqn_no_truth_20260529_205032\checkpoints\worker_best.pt --requester-checkpoint runs\report_full_20260529\platform_dqn_full\platform_dqn_no_truth_20260529_205032\checkpoints\requester_best.pt --output runs\report_full_20260529\platform_dqn_full\platform_dqn_no_truth_20260529_205032\test_eval_best.json`
  - test 基线：`C:\Users\17765\.conda\envs\torch\python.exe -u scripts\run_platform_baselines.py --split test --max-projects 0 --worker-checkpoint runs\report_full_20260529\platform_dqn_full\platform_dqn_no_truth_20260529_205032\checkpoints\worker_best.pt --requester-checkpoint runs\report_full_20260529\platform_dqn_full\platform_dqn_no_truth_20260529_205032\checkpoints\requester_best.pt --output-dir runs\report_full_20260529\platform_baselines_full`
- 模型相关困难/现象：
  - 完整 episode 口径下，train 每轮关闭 1712 个项目，单轮 step 数约 2800 到 3900，显著高于此前 800 步截断设置。
  - 验证集最高 platform reward 出现在 episode 20，但该轮 `requester_hit_rate=0`，说明总 reward 上升并不等于 requester winner 识别能力同步提升。
  - `random_project+wait_until_deadline` 在 test 集仍有异常高 platform reward，但伴随大量 rerouted workers、unfilled projects 和等待成本。
- 调整目的：把报告中的实验表格和曲线统一切换到完整 episode 的正式口径，避免混用 smoke 或截断训练结果。
- 具体调整：不修改模型结构和 reward；只将训练参数从默认 800 步截断改为 `--max-steps 0`，并重新生成报告所需数据。
- 指标变化：
  - 数据统计：`projects=2447`，`entries=186605`，非撤回投稿 `116274`，`workers_with_entries=1753`，`workers_with_quality=1653`，`train_projects=1712`，`val_projects=367`，`test_projects=368`。
  - 训练集最佳：episode 15，`train_platform_reward=661.9700`，`train_worker_hit_rate=0.0771`，`train_requester_hit_rate=0.0040`。
  - 验证集最佳：episode 20，`val_platform_reward=242.0326`，`val_worker_hit_rate=0.0574`，`val_requester_hit_rate=0.0000`，`val_rerouted_workers=173`。
  - test 集 DQN：`platform_reward=265.1064`，`worker_hit_rate=0.05934`，`requester_hit_rate=0.00698`，`winner_quality=0.8499`，`project_wait_cost=1.2482`，`rerouted_workers=205`。
  - test 正常启发式中最高 platform reward 为 `industry_match+worker_industry_match` 的 `231.4816`；DQN 高于这些正常启发式，但低于异常等待基线 `random_project+wait_until_deadline` 的 `913.5873`。
- 可能原因：
  - 完整 episode 使 train reward 中包含完整 project 关闭、等待成本和 worker 回流，reward 尺度较 800 步截断实验明显变大。
  - DQN 的 test platform reward 提升主要来自更多 worker 回流和累计 worker reward，而不是 requester hit 的显著提升。
  - `wait_until_deadline` 的异常高 reward 来自等待到 deadline 后释放大量 worker，导致 worker reward 被重复累计；该策略同时产生 `rerouted_workers=4517`、`unfilled_projects=30` 和 `project_wait_cost=137.9128`，业务上不能简单视为最优。
- 针对本次尝试的改进方向：
  - 对 platform reward 做按 project 平均或按决策步平均，减少回流次数对累计 reward 的放大。
  - 单独提高 requester winner 识别能力，例如加入 winner 监督预训练、提高 winner bonus 或增加申请池质量分布特征。
  - 报告中应同时展示 hit rate、wait cost、unfilled project 和 rerouted workers，避免只按累计 platform reward 排序。

### 2026-05-22 参与者侧全量 Vanilla DQN 初次训练

- 目的：用全量项目训练参与者侧任务推荐模型，获得一版可用于后续 test 评估和 DQN 变体对比的 worker 侧基准模型。
- 数据与设置：全量数据 `max_projects=0`；参与者侧环境；`num_candidates=32`；`include_truth_in_candidates=True`；Vanilla DQN；`device=cuda`；`episodes=10`；每个 episode 最多 800 步。
- 命令：`python scripts/train_worker_dqn.py --max-projects 0 --episodes 10 --device cuda`
- 模型相关困难/现象：
  - 训练集 hit_rate 从 0.05875 上升到 0.10625，但整体仍偏低。
  - 验证集 hit_rate 在第 5 个 episode 达到最高 0.2075，之后回落到 0.15125，存在明显波动。
  - 第 10 个 episode 时 epsilon 仍为 0.810665，探索比例较高，模型尚未充分转向利用已学到的 Q 值。
- 调整目的：本轮未调整模型结构，先建立全量 Vanilla DQN 的初始基线，并观察默认超参下的学习趋势。
- 具体调整：无模型调整；沿用默认奖励、特征、候选集构造、DQN 网络和 epsilon 衰减设置。
- 指标变化：
  - 数据规模：`projects=2447`，`entries=186605`，`train_projects=1712`，`val_projects=367`，`test_projects=368`。
  - 随 episode 变化：

    | Episode | Train Hit@1 | Train Reward | Val Hit@1 | Val Reward | Epsilon | Avg Loss |
    |---------|-------------|--------------|-----------|------------|---------|----------|
    | 1 | 0.05875 | -22.80 | 0.00875 | -71.30 | 0.981665 | 0.0593 |
    | 2 | 0.05125 | -30.48 | 0.13625 | 53.38 | 0.962665 | 0.0403 |
    | 3 | 0.06000 | -23.88 | 0.14125 | 57.70 | 0.943665 | 0.0356 |
    | 4 | 0.05500 | -27.34 | 0.19750 | 113.40 | 0.924665 | 0.0391 |
    | 5 | 0.09250 | 7.86 | 0.20750 | 123.08 | 0.905665 | 0.0405 |
    | 6 | 0.05250 | -30.08 | 0.20625 | 121.86 | 0.886665 | 0.0372 |
    | 7 | 0.06500 | -18.04 | 0.20625 | 121.82 | 0.867665 | 0.0444 |
    | 8 | 0.06875 | -14.62 | 0.20625 | 121.82 | 0.848665 | 0.0421 |
    | 9 | 0.07625 | -7.46 | 0.11750 | 35.64 | 0.829665 | 0.0538 |
    | 10 | 0.10625 | 22.54 | 0.15125 | 68.70 | 0.810665 | 0.0525 |

  - 最佳验证集结果：episode 5，`val_hit_rate=0.2075`，`val_reward=123.08`，保存为 `runs/worker/worker_dqn_20260522_165050/checkpoints/best.pt`。
  - 最终训练集结果：episode 10，`train_hit_rate=0.10625`，`train_reward=22.54`。
  - 最终验证集结果：episode 10，`val_hit_rate=0.15125`，`val_reward=68.70`。
- 可能原因：
  - `train_hit_rate` 小于 `val_hit_rate` 的主要原因是两者统计口径不同：训练时 `run_episode(..., train=True)` 使用 epsilon-greedy 探索，本轮 epsilon 从 0.981665 只降到 0.810665，大量动作仍是随机动作；验证时 `run_episode(..., train=False)` 关闭探索，直接按当前 Q 网络贪心选择动作。因此训练 hit_rate 反映“探索中的行为策略”，验证 hit_rate 反映“当前学到的贪心策略”，二者不能直接理解为训练集表现应高于验证集表现。
  - train 与 val 还来自按项目开始时间切分后的不同事件段，并且每轮只取各自前 800 步；如果这 800 条事件的候选难度、候选数量、worker 历史信息或项目流行度不同，也会造成 val hit_rate 高于 train hit_rate。
  - epsilon 衰减步数为 10000，而本轮仅更新到 global_step 1993，导致后期仍以较高概率随机探索，验证表现容易波动。
  - 每个 episode 限制 800 步，相对于 186605 条 entry 的全量数据仍是子序列训练，覆盖不足可能限制收敛。
  - `include_truth_in_candidates=True` 保证真实项目在候选集内，当前结果主要反映候选集内排序能力，不代表完整召回加排序推荐系统。
- 针对本次尝试的改进方向：
  - 本轮 Vanilla DQN 的 epsilon 仍很高，可以缩短 epsilon 衰减步数或增加训练 episode，让训练后期更多使用 Q 网络而不是随机探索，再观察验证 hit_rate 是否更稳定。
  - 本轮每个 episode 只训练 800 步，可以增加 `max_steps` 或取消步数上限，提高全量事件覆盖率，减少“只学习事件流前段”的偏差。
  - 在保持相同数据和候选集设定下，补充 Double DQN、Dueling DQN 对照，判断 Vanilla DQN 的不足来自网络结构、Q 值估计方式，还是训练步数不足。
  - 针对 `include_truth_in_candidates=True` 做消融，区分“候选集内排序能力”和“真实推荐场景中的召回+排序能力”。

### 2026-05-22 请求者侧全量 Vanilla DQN 初次训练

- 目的：用全量项目训练请求者侧 worker 推荐模型，获得一版可用于后续 test 评估和双端对比的 requester 侧基准模型。
- 数据与设置：全量数据 `max_projects=0`；请求者侧环境；`num_candidates=32`；`include_truth_in_candidates=True`；Vanilla DQN；`device=cuda`；`episodes=10`；每个 episode 最多 800 步。
- 命令：`python scripts/train_requester_dqn.py --max-projects 0 --episodes 10 --device cuda`
- 模型相关困难/现象：
  - 训练集 hit_rate 从 0.03875 上升到 0.1075，但整体仍偏低。
  - 验证集 hit_rate 从 0.18875 上升到第 8 个 episode 的最高 0.23625，之后略有回落。
  - 训练 hit_rate 仍明显低于验证 hit_rate，且第 10 个 episode 时 epsilon 仍为 0.810665，训练行为中随机探索占比仍高。
- 调整目的：本轮不修改模型设定，先建立请求者侧全量 Vanilla DQN 初始基线，观察默认奖励、特征和候选策略下的学习趋势。
- 具体调整：无模型调整；沿用默认奖励、特征、候选集构造、DQN 网络和 epsilon 衰减设置。
- 指标变化：
  - 数据规模：`projects=2447`，`entries=186605`，`train_projects=1712`，`val_projects=367`，`test_projects=368`。
  - 随 episode 变化：

    | Episode | Train Hit@1 | Train Reward | Val Hit@1 | Val Reward | Epsilon | Avg Loss |
    |---------|-------------|--------------|-----------|------------|---------|----------|
    | 1 | 0.03875 | -38.46 | 0.18875 | 123.56 | 0.981665 | 0.0383 |
    | 2 | 0.04875 | -27.88 | 0.19500 | 130.58 | 0.962665 | 0.1372 |
    | 3 | 0.07500 | 0.80 | 0.22375 | 161.79 | 0.943665 | 0.1315 |
    | 4 | 0.06250 | -14.52 | 0.21625 | 155.14 | 0.924665 | 0.1281 |
    | 5 | 0.08000 | 5.11 | 0.19875 | 135.43 | 0.905665 | 0.1783 |
    | 6 | 0.07875 | 4.57 | 0.21625 | 156.09 | 0.886665 | 0.1743 |
    | 7 | 0.09500 | 23.13 | 0.21875 | 158.21 | 0.867665 | 0.1396 |
    | 8 | 0.10500 | 30.31 | 0.23625 | 174.42 | 0.848665 | 0.1853 |
    | 9 | 0.09000 | 15.33 | 0.22750 | 166.15 | 0.829665 | 0.1749 |
    | 10 | 0.10750 | 34.14 | 0.21750 | 155.29 | 0.810665 | 0.1563 |

  - 最佳验证集结果：episode 8，`val_hit_rate=0.23625`，`val_reward=174.42`，保存为 `runs/requester/requester_dqn_20260522_172331/checkpoints/best.pt`。
  - 最终训练集结果：episode 10，`train_hit_rate=0.1075`，`train_reward=34.14`。
  - 最终验证集结果：episode 10，`val_hit_rate=0.2175`，`val_reward=155.29`。
- 可能原因：
  - 与参与者侧相同，训练 hit_rate 统计的是高 epsilon 探索下的行为策略；验证 hit_rate 统计的是关闭探索后的贪心策略，因此训练 hit_rate 低于验证 hit_rate 是合理现象。
  - 请求者侧候选 worker 由质量分和历史活动排序补齐，默认还会把真实投稿 worker 放入候选集，因此验证指标主要衡量候选集内排序能力。
  - 验证 hit_rate 相比参与者侧略高，可能与请求者侧候选 worker 的质量分、历史活跃度特征更直接相关有关；该判断仍需通过 test 集和基线对比确认。
  - 训练仍只覆盖每个 episode 前 800 步，且 epsilon 衰减不足，模型尚未充分收敛。
- 针对本次尝试的改进方向：
  - 请求者侧 Vanilla DQN 的验证 hit_rate 已高于简单基线预期，可以重点比较 Double DQN、Dueling DQN 是否能进一步提升候选 worker 排序能力。
  - 当前 reward 同时包含命中、分数、worker quality、winner/finalist，加权方式仍是经验设定；可以做奖励权重消融，比较更强调质量分或更强调投稿分数时 Hit@1 与 reward 的变化。
  - 当前候选 worker 由质量分和历史活动排序补齐，可以增加候选集构造消融，例如降低质量分排序强度或引入更多随机负样本，检查模型是否真正学到项目-worker 匹配，而不是只复用候选生成规则。
  - 和参与者侧一样，可以增加训练步数或加快 epsilon 衰减，观察验证 hit_rate 是否更稳定。

### 2026-05-22 双端全量 Dueling + Double DQN 训练与 test 评估

- 目的：在 Vanilla DQN 基线之后，测试 DQN 系列增强版本 `Dueling + Double DQN` 在参与者侧和请求者侧的表现，并与 test 集基线方法对比。
- 数据与设置：全量数据 `max_projects=0`；`num_candidates=32`；`include_truth_in_candidates=True`；`model_type=dueling`；`double_dqn=True`；`episodes=10`；每个 episode 最多 800 步。
- 命令：
  - `python scripts/train_worker_dqn.py --max-projects 0 --episodes 10 --device cuda --model dueling --double-dqn`
  - `python scripts/train_requester_dqn.py --max-projects 0 --episodes 10 --device cuda --model dueling --double-dqn`
  - `python scripts/run_baselines.py --side worker --split test --max-projects 0 --checkpoint runs/worker/worker_dqn_20260522_173431/checkpoints/best.pt --output-dir runs/baselines_dueling_double`
  - 请求者侧 test 对比结果保存到 `runs/baselines_dueling_double/requester_test/comparison.csv`。
- 模型相关困难/现象：
  - 参与者侧验证 hit_rate 在 episode 5 达到最高 0.22375，但之后大幅波动，episode 9 降到 0.03125，episode 10 回升到 0.21000。
  - 请求者侧验证 hit_rate 在 episode 5 达到最高 0.25625，之后维持在 0.19 到 0.20 左右。
  - 参与者侧 test 中 Dueling + Double DQN 的 Hit@1 为 0.09519，明显低于 `category_match` 的 0.95143。
  - 请求者侧 test 中 Dueling + Double DQN 的 Hit@1 为 0.28873，明显高于所有请求者侧基线。
- 调整目的：通过 Dueling 网络分离状态价值和动作优势，并用 Double DQN 降低 Q 值过估计，观察是否提升候选集内排序能力。
- 具体调整：
  - Q 网络从 Vanilla DQN 切换为 Dueling Q Network。
  - TD target 从普通 DQN 切换为 Double DQN：用 policy net 选择 next action，用 target net 估计该 action 的 Q 值。
  - 其他训练设置保持与 Vanilla DQN 相同，便于对比。
- 指标变化：
  - 参与者侧训练/验证随 episode 变化：

    | Episode | Train Hit@1 | Train Reward | Val Hit@1 | Val Reward | Epsilon | Avg Loss |
    |---------|-------------|--------------|-----------|------------|---------|----------|
    | 1 | 0.04875 | -34.02 | 0.02875 | -52.06 | 0.981665 | 0.0514 |
    | 2 | 0.07000 | -13.76 | 0.22000 | 135.06 | 0.962665 | 0.0401 |
    | 3 | 0.07000 | -13.52 | 0.21875 | 134.00 | 0.943665 | 0.0408 |
    | 4 | 0.06000 | -23.48 | 0.22250 | 137.54 | 0.924665 | 0.0424 |
    | 5 | 0.07875 | -5.14 | 0.22375 | 138.64 | 0.905665 | 0.0439 |
    | 6 | 0.07250 | -11.04 | 0.08000 | -2.42 | 0.886665 | 0.0387 |
    | 7 | 0.06625 | -16.56 | 0.16625 | 83.44 | 0.867665 | 0.0433 |
    | 8 | 0.06375 | -19.94 | 0.20250 | 118.18 | 0.848665 | 0.0424 |
    | 9 | 0.07375 | -8.88 | 0.03125 | -49.58 | 0.829665 | 0.0461 |
    | 10 | 0.06875 | -13.76 | 0.21000 | 125.40 | 0.810665 | 0.0466 |

  - 请求者侧训练/验证随 episode 变化：

    | Episode | Train Hit@1 | Train Reward | Val Hit@1 | Val Reward | Epsilon | Avg Loss |
    |---------|-------------|--------------|-----------|------------|---------|----------|
    | 1 | 0.04375 | -32.13 | 0.16875 | 103.88 | 0.981665 | 0.0430 |
    | 2 | 0.05875 | -16.32 | 0.14875 | 81.48 | 0.962665 | 0.1172 |
    | 3 | 0.05000 | -25.75 | 0.12625 | 55.94 | 0.943665 | 0.1011 |
    | 4 | 0.06375 | -11.78 | 0.19125 | 127.04 | 0.924665 | 0.1395 |
    | 5 | 0.06500 | -10.57 | 0.25625 | 195.01 | 0.905665 | 0.1018 |
    | 6 | 0.08000 | 5.38 | 0.19625 | 130.64 | 0.886665 | 0.1149 |
    | 7 | 0.08625 | 12.78 | 0.20125 | 136.81 | 0.867665 | 0.1340 |
    | 8 | 0.08000 | 5.28 | 0.20375 | 139.48 | 0.848665 | 0.1487 |
    | 9 | 0.10625 | 31.79 | 0.19250 | 126.45 | 0.829665 | 0.1472 |
    | 10 | 0.10250 | 30.80 | 0.19750 | 132.95 | 0.810665 | 0.2660 |

  - 参与者侧 test 对比：

    | Policy | Hit@1 | Reward | Hits / Steps |
    |--------|-------|--------|--------------|
    | random | 0.03517 | -1026.04 | 614 / 17459 |
    | popularity | 0.07349 | -275.82 | 1283 / 17459 |
    | category_match | 0.95143 | 17734.92 | 16611 / 17459 |
    | award | 0.08013 | -121.00 | 1399 / 17459 |
    | Dueling + Double DQN | 0.09519 | 175.30 | 1662 / 17459 |

  - 请求者侧 test 对比：

    | Policy | Hit@1 | Reward | Hits / Steps |
    |--------|-------|--------|--------------|
    | random | 0.03122 | -1030.45 | 545 / 17459 |
    | worker_quality | 0.05779 | -382.95 | 1009 / 17459 |
    | worker_activity | 0.03110 | -1025.36 | 543 / 17459 |
    | Dueling + Double DQN | 0.28873 | 4857.18 | 5041 / 17459 |

- 可能原因：
  - 请求者侧的 worker quality、历史活跃度和投稿分数等特征与真实投稿 worker 的相关性较强，Dueling + Double DQN 能在候选 worker 内学到比简单基线更有效的排序规则。
  - 参与者侧 `category_match` 极高，核心原因是当前候选集与特征设计让“类目匹配”变成了接近 oracle 的规则：`include_truth_in_candidates=True` 会把真实投稿项目放进候选集，且真实项目在候选列表中优先加入；`category_match` 从前往后选择第一个与 worker 主导类目一致的项目，只要真实项目类目等于 worker 主导类目，就很容易直接选中真实项目。
  - worker 主导类目当前由 worker 的全量历史投稿统计得到，而不是严格只用当前时间之前的历史。这会让类目偏好特征包含未来信息，尤其在 test 阶段会强化 `category_match`，使它不完全是一个真实可部署的在线推荐基线。
  - DQN 虽然也能看到项目特征中的 `cat_match` 信号，但它没有利用“真实项目被优先放入候选列表”这一规则；当多个候选项目都与主导类目匹配时，DQN 还需要依靠奖金、热度、时间等连续特征排序，因此不一定能复制 `category_match` 的强位置偏置。
  - Dueling + Double DQN 在参与者侧验证曲线波动较大，可能说明 10 个 episode、每轮 800 步仍不足以稳定学习；也可能是候选项目排序中类目信号过强，DQN 的优势主要体现在非类目匹配样本上。
  - 两侧训练阶段 epsilon 仍较高，训练 hit_rate 继续低于验证/test 贪心策略表现，口径原因与前两次 Vanilla DQN 相同。
- 针对本次尝试的改进方向：
  - 先修正参与者侧评估口径：增加 `include_truth_in_candidates=False`，或保留真实项目但将候选顺序随机打乱，避免真实项目固定优先进入候选列表导致 `category_match` 借到位置优势。
  - 修正 worker 类目偏好特征：把全量主导类目改为“仅基于当前时间之前历史投稿”的主导类目，并增加过去 Top-N 类目分布、子类目偏好、行业偏好、近期类目偏好等过去信息特征，减少未来信息泄漏。
  - 增加对 DQN 更友好的匹配特征：例如项目类目是否在 worker 历史 Top-3 类目中、worker 在该类目下的历史平均分/获奖率、距离上次同类目投稿的时间、项目奖金与 worker 历史偏好的交互项。这样模型不只看到单一 `cat_match`，而能区分多个同类目候选项目。
  - 调整参与者侧 reward：当前 reward 以命中真实投稿为主，可以加入更贴近参与者利益的 shaping，例如项目奖金、项目质量/平均分、worker 在该类目的历史成功率、截止时间压力等，让模型学习“值得推荐”的任务，而不只是复制历史选择。
  - 加入监督式辅助目标或预训练：在 DQN 更新外增加候选项目 Hit@1 的交叉熵/排序损失，先让网络学会从候选集中识别真实项目，再用 DQN reward 微调，可能缓解早期高 epsilon 和稀疏命中奖励导致的学习慢。
  - 对报告解释要分两套结果：一套说明当前 `include_truth_in_candidates=True` 下是候选集内排序实验，另一套用修正后的候选集和过去信息特征说明更接近真实推荐场景的表现。
