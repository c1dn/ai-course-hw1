# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python educational homework assignment for implementing Go (围棋) AI algorithms based on the book "Deep Learning and the Game of Go". Students implement MCTS (Monte Carlo Tree Search) and Minimax with Alpha-Beta pruning algorithms.

## PDF 作业要求（三小问）

### 第一小问（必做）：随机 AI
- **文件**: `agents/random_agent.py` ✅ 已创建
- **要求**: 5×5 棋盘，随机合法落子，验证规则调用
- **测试**: `python play.py --agent1 random --agent2 random --size 5`

### 第二小问（必做）：MCTS + 优化
- **文件**: `agents/mcts_agent.py`
- **基础**: 标准 MCTS（选择、扩展、模拟、反向传播）
- **关键**: **必须实现至少两种优化方法**：
  1. 启发式走子策略（非完全随机）
  2. 限制模拟深度（如 20-30 步）
  3. 其他：RAVE、池势等
- **时间**: 每步 ≤ 10 秒

### 第三小问（选做）：Minimax
- **文件**: `agents/minimax_agent.py`
- **要求**: Alpha-Beta 剪枝，与 MCTS 对比
- **注意**: 标记为【选做】，如果时间紧张可跳过

## Project Structure

```
hw1/
├── dlgo/                  # 【资产】基础设施（已完成，勿动）
│   ├── gotypes.py         # Player enum, Point namedtuple
│   ├── goboard.py         # Board, GameState (immutable), Move
│   ├── scoring.py         # Territory scoring, GameResult
│   └── zobrist.py         # Precomputed Zobrist hash table (19x19)
│
├── agents/                # 【学生实现】按PDF三小问完成
│   ├── random_agent.py    # ⭐ 第一小问（必做）：RandomAgent
│   ├── mcts_agent.py      # ⭐ 第二小问（必做）：MCTSAgent + 两种优化
│   └── minimax_agent.py   # ⭐ 第三小问（选做）：MinimaxAgent
│
└── play.py                # 对弈脚本（已适配三小问）
```

## Architecture Notes

### Immutable Game State Design
- `GameState` is immutable - `apply_move()` returns a new state
- Uses `copy.deepcopy()` for board copies
- `previous_states` frozenset tracks history for ko detection
- `Board` uses Zobrist hashing for O(1) position comparison

### Key Classes
- `Player`: Enum with `black`, `white`, and `other` property
- `Point`: Namedtuple with `row`, `col` (1-indexed), `neighbors()` method
- `Move`: Created via `Move.play(point)`, `Move.pass_turn()`, `Move.resign()`
- `GameState`: Entry point via `GameState.new_game(board_size)`

### Ko Detection
- Uses Zobrist hash stored in `previous_states` frozenset
- `does_move_violate_ko()` checks if resulting position repeats

## Common Commands

### Test Infrastructure
```bash
python -c "from dlgo import GameState; g = GameState.new_game(9); print('OK:', g.board.num_rows)"
```

### Run a Single Game
```bash
# Two random agents
python play.py --agent1 random --agent2 random --size 5

# Test MCTS implementation
python play.py --agent1 mcts --agent2 random --size 5 --rounds 50

# Test Minimax implementation
python play.py --agent1 minimax --agent2 random --size 9 --depth 3

# MCTS vs Minimax
python play.py --agent1 mcts --agent2 minimax --size 5 --games 10
```

### Quick Test Agent Implementation
```bash
# Test MCTSAgent
python -c "
from dlgo import GameState
from agents.mcts_agent import MCTSAgent
game = GameState.new_game(5)
agent = MCTSAgent(num_rounds=100)
move = agent.select_move(game)
print('MCTS selected:', move)
"

# Test MinimaxAgent
python -c "
from dlgo import GameState
from agents.minimax_agent import MinimaxAgent
game = GameState.new_game(5)
agent = MinimaxAgent(max_depth=3)
move = agent.select_move(game)
print('Minimax selected:', move)
"
```

## Implementation Requirements

### MCTS Agent (`agents/mcts_agent.py`)
5 TODOs to implement:
1. `MCTSNode.best_child()` - UCT selection formula
2. `MCTSNode.expand()` - Expand child nodes for legal moves
3. `MCTSNode.backup()` - Backpropagate simulation results
4. `MCTSAgent.select_move()` - Main MCTS loop (select/expand/simulate/backup)
5. `MCTSAgent._simulate()` - Random rollout with **at least 2 optimization methods**
   - 启发式走子策略 (heuristic playout policy)
   - 限制模拟深度 (depth-limited simulation)
   - 其他可选: RAVE, pattern-based moves, etc.

### Minimax Agent (`agents/minimax_agent.py`) - 【选做】
4 TODOs to implement (optional, for comparison with MCTS):
1. `minimax()` - Basic recursive minimax algorithm
2. `alphabeta()` - Alpha-beta pruning optimization
3. `_default_evaluator()` - Position evaluation heuristic
4. `GameResultCache.put()` - Transposition table caching

## Scoring

- Territory scoring (area scoring): stones + territory
- Komi: 7.5 points for white
- `compute_game_result(game_state)` returns `GameResult` with `winner` property
