# 围棋 AI 课程项目

使用了提供的框架代码：<https://github.com/zhenqis123/ai-course-hw1>
报告 PDF：[`report/report.pdf`](report/report.pdf)

本项目在课程给定的围棋规则框架之上，实现了一个可直接运行的围棋 AI 系统。GUI 保留了 5 / 7 / 9 棋盘入口，实验主要在 5x5 棋盘上进行。

当前实现包含：

- `RandomAgent`：合法随机落子，用于验证规则调用。
- `MCTSAgent`：标准 MCTS 流程 + 启发式扩展、启发式 rollout、prior bonus、rollout 截断静态评估。
- `MinimaxAgent`：选做实现，包含 Alpha-Beta 剪枝、轻量置换表、棋串级静态评估和走子排序。
- `PySide6` 图形界面：支持 Human vs AI / AI vs AI、参数设置、Pass / Undo / Resign、日志显示。
- 参数化实验脚本：支持单配置实验、cross-play 批量实验、1000 rounds 的 MCTS 消融实验。

## 项目结构

```text
ai-course-hw1/
├── agents/
│   ├── random_agent.py
│   ├── mcts_agent.py
│   ├── minimax_agent.py
│   └── policy/
│       ├── opening_policy.py
│       ├── mcts_policy.py
│       └── minimax_policy.py
├── dlgo/                         # 围棋规则、计分、Zobrist 哈希
├── experiments/
│   ├── run_experiments.py        # 单配置实验入口
│   ├── run_experiment_plan.py    # 批量实验计划入口
│   └── results/
│       ├── experiment_plan/      # MCTS vs Minimax 正式结果
│       └── mcts_ablation_plan/   # 1000 rounds MCTS 消融结果
├── gui/
├── report/
│   ├── report.tex
│   ├── generate_figures.py
│   └── figures/
├── docs/
│   └── homework.pdf
├── play.py
├── play_gui.py
└── requirements.txt
```

## 环境安装

```bash
python -m pip install -r requirements.txt
```

当前主要外部依赖是 `PySide6`。实验与命令行对弈主要依赖 Python 标准库。

## 运行方式

### 1. 命令行对弈

```bash
# MCTS vs Random
python play.py --agent1 mcts --agent2 random --size 5

# Minimax vs MCTS
python play.py --agent1 minimax --agent2 mcts --size 5 --games 10 --quiet
```

说明：

- `agent1` 为黑方，`agent2` 为白方。
- `play.py` 是轻量入口，内部固定使用：
  - `MCTS(num_rounds=100)`
  - `Minimax(max_depth=3)`
- 如果需要精确控制 rounds / depth，请使用实验脚本。

### 2. 图形界面

```bash
python play_gui.py
```

GUI 支持：

- 模式：`Human vs AI`、`AI vs AI`
- 棋盘：`5 / 7 / 9`
- 智能体：`Random / MCTS / Minimax`
- 参数：`MCTS rounds`、`Minimax depth`
- 操作：`Pass / Undo / Resign`
- 状态查看：回合、结果、步数、分数估计、走子日志

与代码实现强相关的两个细节：

- GUI 中 `MCTS rounds = -1` 表示 `auto rounds`，不是无限搜索。
- 在 `5x5` 棋盘上，这个 `auto rounds` 等效为 `600 rounds`。
- 当 GUI 中 rounds 为正数时，MCTS 会使用 `max_rollout_depth=28`。
- 当 rounds 为 `-1` 时，rollout 走到终局。

### 3. 单配置实验

```bash
python experiments/run_experiments.py \
  --black-agent mcts \
  --white-agent minimax \
  --mcts-rounds 2200 \
  --minimax-depth 5 \
  --games 10 \
  --output-dir experiments/results/demo_mcts2200_vs_minimax5
```

若双方都使用 MCTS，可分别传入黑白侧参数，例如：

```bash
python experiments/run_experiments.py \
  --black-agent mcts \
  --white-agent mcts \
  --black-mcts-rounds 1000 \
  --white-mcts-rounds 1000 \
  --black-mcts-exploration-weight 1.414 \
  --white-mcts-exploration-weight 0.6 \
  --black-mcts-max-rollout-depth 50 \
  --white-mcts-max-rollout-depth 10 \
  --black-mcts-rollout-policy random \
  --white-mcts-rollout-policy heuristic \
  --black-mcts-expansion-policy uniform \
  --white-mcts-expansion-policy heuristic \
  --black-mcts-use-prior-bonus false \
  --white-mcts-use-prior-bonus true \
  --games 10 \
  --output-dir experiments/results/demo_ablation
```

输出目录通常包含：

- `per_game.csv`：逐局结果
- `summary.csv`：配置级汇总
- `summary.json`：包含元数据的汇总结果
- `game_seeds.csv`：随机种子列表

### 4. 批量实验计划

```bash
# MCTS vs Minimax
python experiments/run_experiment_plan.py --plan crossplay --skip-existing

# 1000 rounds MCTS 消融
python experiments/run_experiment_plan.py --plan ablation --skip-existing
```

## 实验使用的规则

- 实验使用 `5x5` 棋盘。
- 默认贴目为 `7.5`，来自 `dlgo/scoring.py` 的 `default_komi_for_board_size()`。
- 5x5 空棋盘黑棋第一手固定走天元 `(3, 3)`，所有 Agent 共用该开局策略。（因为已经被证明是5x5 棋盘上最优的。）
- 自然终局由连续两手 `pass` 或一方 `resign` 触发；计分采用“子数 + 地盘”，白方额外加贴目。
- 实验脚本默认 `move_limit = board_size * board_size * 2`，因此 5x5 为 50 手；达到 50 手仍未自然终局时，按当前局面计分裁定，记为 `move_limit_score`。
- `--move-time-limit-s` 只用于记录是否出现慢步，不会中断搜索。
- MCTS vs Minimax 计划中，前 5 局 MCTS 执黑，后 5 局 MCTS 执白；MCTS 消融计划中，前 5 局标准 MCTS 执黑、变体执白，后 5 局反过来。

## 实现要点

- `MCTSAgent` 同时支持：
  - 标准MCTS `standard_baseline()`
  - 当前实验使用的增强配置
- 当前 cross-play 实验中的增强版 MCTS 默认配置为：
  - `exploration_weight=0.6`
  - `max_rollout_depth=10`
  - `rollout_policy=heuristic`
  - `expansion_policy=heuristic`
  - `use_prior_bonus=True`
- 当前标准 MCTS 消融对照配置为：
  - `num_rounds=1000`
  - `exploration_weight=1.414`
  - `max_rollout_depth=50`
  - `rollout_policy=random`
  - `expansion_policy=uniform`
  - `use_prior_bonus=False`

## 实验结果摘要

完整分析见 [`report/report.pdf`](report/report.pdf)。核心结果如下：

- MCTS 消融实验中，`Cutoff-10 + eval` 单项优化在 10 局中胜率为 100%，且平均每步耗时显著低于标准 MCTS。
- `Full optimized` MCTS 在 1000 rounds 消融中胜率为 90%，平均胜负差最大，综合棋力和速度表现最好。
- MCTS vs Minimax 的同步预算实验中，`MCTS(300) vs Minimax(4)`、`MCTS(600) vs Minimax(5)`、`MCTS(2200) vs Minimax(6)` 的 MCTS 胜率均为 0%。
- 固定 `Minimax(depth=5)` 时，MCTS 胜率随 rounds 变化为：2200 rounds 为 0%，3000 rounds 为 10%，4000 rounds 为 50%，5000 rounds 为 40%。
- 高`num_rounds` MCTS 的胜局几乎全部来自执白，说明结果受到贴目、颜色和 50 手裁定机制的明显影响。

## AI辅助声明

本项目在代码整理、实验中使用了 `Codex` 作为辅助工具。AI 主要用于：

- 编写GUI
- 整理策略为策略库
- 编写代码注释
- 生成批量实验脚本
- 生成实验结果图表
