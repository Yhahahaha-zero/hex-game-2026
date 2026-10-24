# 页面层与 HexEngine 接口契约

本文档用于约定页面层与规则引擎的职责边界。页面版本稳定后，应按本文档统一接入 `HexEngine`。

## 1. 核心原则

- `HexEngine` 是规则、回合、判胜、计分和历史记录的唯一来源。
- 页面只负责显示、接收用户输入、播放动画和调度 AI。
- 页面不得直接调用 `board.place()` 完成正式落子。
- 页面不得自行调用 `has_won()`、`winning_path()` 决定比赛结果。
- 页面不得自行交换红蓝、累计比分或切换当前回合。
- AI 只负责返回着法，不能直接修改棋盘或计分。

## 2. 正式接口

| 接口 | 返回值 | 说明 |
|---|---|---|
| `start_match()` | `None` | 开始或重开一场比赛，清空比分和历史 |
| `play_move(q, r, ms=0.0, reason=None)` | `bool` | 当前方落子；本手获胜返回 `True` |
| `next_game()` | 新状态字符串 | 本局结束后进入下一局；第二局后进入 `match_over` |
| `current_player` | 队伍名或 `None` | 当前执棋队伍；非 `playing` 状态返回 `None` |
| `current_color` | `RED`、`BLUE` 或 `None` | 当前执棋颜色；非 `playing` 状态返回 `None` |
| `winner` | 队伍名或 `None` | 本局胜方；未产生胜者时为 `None` |
| `match_score` | `dict` 快照 | 按队伍统计的胜局数 |
| `move_history` | `tuple` 快照 | 整场比赛的手数记录 |
| `state` | 字符串 | `playing`、`game_over`、`match_over` |
| `game_no` | `1` 或 `2` | 当前局号 |
| `board` | `HexBoard` | 只读显示用途；正式落子必须走 `play_move()` |
| `winner_color` | `RED`、`BLUE` 或 `None` | 本局胜方颜色 |
| `win_path` | 坐标序列或 `None` | 获胜通路，供页面高亮 |
| `end_reason` | 字符串或 `None` | `connect`、超时、非法落子等结束原因 |
| `match_winner()` | 队伍名或 `None` | 仅在 `match_over` 时有效；1:1 返回 `None` |

旧接口 `reset_match()`、`play()` 和 `turn` 仍保留兼容，但新页面应使用正式接口。

## 3. 状态流转

```text
start_match()
    -> playing
       -> play_move() 未获胜 -> playing
       -> play_move() 获胜   -> game_over
    -> next_game()
       -> 第 1 局结束后 -> playing（交换红蓝）
       -> 第 2 局结束后 -> match_over
```

每局固定由红方先行。第二局开始时双方队伍交换红蓝，但 `current_color` 仍从 `RED` 开始。

## 4. 职责划分

### HexEngine 负责

- 校验落子是否合法；
- 维护当前执棋颜色和队伍；
- 自动切换回合；
- 判断胜负；
- 生成获胜通路；
- 统计两局比分；
- 第二局交换红蓝；
- 保存完整手数和历史；
- 拒绝结束后继续落子或重复裁定。

### 页面层负责

- 绘制棋盘、棋子、按钮和提示；
- 将鼠标坐标转换为 `(q, r)`；
- 将用户操作转成 `play_move(q, r)`；
- 根据 `current_player` 决定当前是人类还是 AI；
- 根据 `current_color` 让 AI 选择颜色；
- 播放落子和胜利动画；
- 显示 `IllegalMoveError` 等错误信息；
- 根据 `state` 显示开始、对局、胜利或本场结束页面。

### AI 负责

- 根据 `board`、`current_color` 和规则返回 `((q, r), reason)`；
- 不得直接调用 `play_move()`；
- 不得修改棋盘；
- 必须保证返回空位或抛出明确异常；
- 模型加载、版本和耗时由 AI 层自行记录。

## 5. 页面集成流程

```python
from hex_board import RED, BLUE, IllegalMoveError
from hex_engine import HexEngine

engine = HexEngine("玩家1", "玩家2")
engine.start_match()

# 用户点击或 AI 决策后：
try:
    game_won = engine.play_move(q, r, ms=think_ms, reason=reason)
except IllegalMoveError as exc:
    show_error(str(exc))
else:
    refresh_view(engine)
    if engine.state == "game_over":
        show_winner(engine.winner, engine.win_path)

# 进入下一局：
if engine.state == "game_over":
    engine.next_game()

# 本场结束后重新开始：
if engine.state == "match_over":
    engine.start_match()
```

页面绘制时应读取：

```python
engine.board.grid
engine.current_color
engine.current_player
engine.match_score
engine.move_history
engine.win_path
engine.state
```

## 6. 手数记录格式

`move_history` 中每条记录的结构：

```python
{
    "game": 1,
    "no": 1,
    "color": RED,
    "side": "玩家1",
    "cell": "6F",
    "q": 5,
    "r": 5,
    "ms": 12.35,
    "reason": "开局",
}
```

`match_score` 和 `move_history` 返回的是快照，页面修改返回值不会影响引擎内部状态。

## 7. 接入验收清单

- 页面中不存在直接调用 `board.place()` 完成正式落子的代码；
- 页面中不存在自行判断胜负的代码；
- 页面中不存在手动切换回合的代码；
- 第二局能够正确交换双方执色；
- 两局比分由 `match_score` 显示；
- 结束后继续落子会捕获 `IllegalMoveError`；
- 重开比赛会调用 `start_match()`；
- GUI 冒烟测试和 `test_hex_engine.py` 同时通过。

## 8. 当前集成状态

`HexEngine` 统一接口和单元测试已经完成。`hex_game.py` 暂不修改，等待页面队友版本稳定后再按本契约接入。
