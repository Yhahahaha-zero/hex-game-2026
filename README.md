# 海克斯棋（Hex）

双人落子类游戏。11×11 六边形棋盘，红方连通上下红色边界获胜，蓝方连通左右蓝色边界获胜。单局无和棋，每场两局、双方轮流先手（红方每局先行）。坐标先纵后横，如 `5C` = 第 5 行第 C 列。

## 环境准备（只需一次）

1. 安装 Python 3（python.org 下载，安装时勾选 "Add python.exe to PATH"）；
2. 安装 pygame：

```bash
py -m pip install pygame      # Python 3.13 及以下
py -m pip install pygame-ce   # Python 3.14 及以上（导入名同为 pygame，代码无需改动）
```

`start_game.bat` 会自动完成第 2 步（自动识别 Python 版本选择正确的包）。

### 神经网络 AI（可选）

使用神经 MCTS 时安装 AI 依赖：

```bash
python -m pip install -r requirements-ai.txt
```

默认模型路径为 `checkpoints/hex_model_v5.pt`。当前附带的是训练 10 局的集成验证参考模型，仅用于跑通流程，不能代表最终比赛水平；正式提交前应替换为队友的完整训练权重。模型或 torch 不可用时，人机模式会自动回退基础 AI，双人模式不受影响。模型说明见 `checkpoints/README.md`。

???? AI ?? `HybridAI`?Dijkstra + ? + ?? MCTS????????????

```powershell
$env:HEX_AI_ENGINE = "neural"
```

???? `hybrid`?`neural`?`basic`?


## 常见问题

**Q: 启动时报 `Could not find a version that satisfies the requirement pygame (from versions: none)`？**

- 最常见原因：Python 是 **3.14 或更新**，经典 pygame 没有对应安装包。新版 `start_game.bat` 会自动改用 pygame-ce；手动安装命令：
  ```bash
  py -m pip install pygame-ce
  ```
- 若确认 Python 是 3.13 及以下，可能是 pip 镜像源异常，指定官方源重试：
  ```bash
  py -m pip install pygame -i https://pypi.org/simple
  ```
- 查看 Python 版本：`py --version`；建议使用 Python 3.11 ~ 3.13。

## 启动游戏

任选其一：

- **双击 `start_game.bat`**（推荐，无需打开命令行）
- 或在本文件夹打开命令行执行：`py hex_game.py`

### 环境自检（可选）

```bat
start_game.bat --check
```

脚本会自动寻找带 pip 的 Python，并检查 pygame 是否可用；此模式不会安装依赖或启动游戏。

### 开始画面按数字键 1-4（或点击按钮）选择模式

| 按键 | 模式 | 说明 |
|---|---|---|
| 1 | 双人对战 | 两名玩家轮流点击落子 |
| 2 | 人机对战 · 玩家执红 | 玩家先手 |
| 3 | 人机对战 · 玩家执蓝 | AI 先手 |
| 4 | AI 对 AI 观战 | 空格暂停/继续，本局结束自动进下一局 |

### 操作

- 鼠标点击空六边形落子；悬停时左下角显示坐标（先纵后横）
- 空格：开始 / 继续 / 下一局 / 观战暂停
- 右侧面板实时显示红蓝双方计时和最近手数
- `Ctrl+S`：保存当前棋局为 JSON
- `F5`：加载最近记录进入复盘；方向键切换手数，Esc 返回
- `Ctrl+E`??? JSON?CSV?MCTS ??????????
- `Ctrl+P`???????????
- Esc：退出
- 胜负判定：红方连通上下边界、蓝方连通左右边界即胜，获胜通路以金圈高亮

## AI 对 AI 自动对局工具（可选）

```bash
py hex_match.py                          # 100 局 基础AI vs 基础AI
py hex_match.py --games 500              # 指定局数
py hex_match.py --ai-red random --ai-blue basic   # 指定双方 AI
py hex_match.py --list                   # 列出对局记录
py hex_match.py --replay 7               # 逐步复现第 7 局（回车走一步）
py hex_match.py --replay 7 --speed 0.3   # 每 0.3 秒自动一步
py hex_match.py --replay-lost red        # 依次复现红方失败的全部棋局
```

统计输出包括：胜率（红/蓝）、平均手数、运行总耗时、AI 单步思考平均与最大值；对局记录（每手坐标、AI 决策原因、耗时）保存于 `hex_records.json`。

## 运行自测


一键检查（推荐）：

```bat
check.bat
```

也可单独运行以下命令：
```bash
py hex_game_test.py
py -m unittest -v test_hex_board.py test_hex_engine.py test_hex_match.py test_benchmark_ai.py test_ai.py test_mcts.py test_ai_worker.py test_ui.py test_p1.py test_p2.py test_evaluation.py test_tactics.py test_hybrid_ai.py
```

## AI 对比评测

接口、模型文件和回退规则见 `docs/AI_INTERFACE.md`。

固定随机种子并采用 AB/BA 配对交换红蓝：

```bash
python tools/benchmark_ai.py --ai-a basic --ai-b random --games 500 --repeats 3
```


神经 AI 对基础 AI：

```bash
python tools/benchmark_ai.py --ai-a basic --ai-b neural --games 1000 --repeats 3 --seed 20261024
```

检查模型文件和单步耗时：

```bash
python tools/check_model.py checkpoints/hex_model_v5.pt
```

训练 V5 模型：

```bash
python ai/train_v5.py --games 500 --seed 20261024 --output checkpoints/hex_model_v5.pt
```

训练完成后会同时生成 `model_config.json` 和 `.sha256` 校验文件。
自定义 AI 使用 `module:Class` 或 `path/to/file.py:Class`。结果写入 `benchmark_results/`，包含 JSON、汇总 CSV 和逐局 CSV。


## 文件说明

| 文件 | 职责 |
|---|---|
| `hex_game.py` | pygame 图形界面（主程序） |
| `hex_board.py` | 规则核心：棋盘、BFS/DFS 连通性判定、并查集快速判定、文本棋盘 |
| `hex_ai.py` | AI：直接获胜 > 紧急防守 > 中心偏好随机；预留路径评估扩展点 |
| `ai/neural_ai.py` | V5 神经网络、战术预检查、有限时间 MCTS 和回退工厂 |
| `ai/mcts.py` | PUCT MCTS、合法动作掩码和单步时间上限 |
| `ai/model_v5.py` | HexNetV5 网络结构和动作空间工具 |
| `ai/train_v5.py` | 可复现 V5 自对弈训练与模型元数据保存 |
| `ai/evaluation.py` | Dijkstra ?????????????? |
| `ai/tactics.py` | ????????????? |
| `ai/hybrid_ai.py` | ?????? AI |
| `ai/ai_factory.py` | ?? AI ????????? |
| `docs/AI_EVALUATION.md` | HybridAI ? NeuralMCTS ???? |
| `ai/worker.py` | 后台 AI 线程、回退策略和 AIResult |
| `ui/board_renderer.py` | 棋盘几何、棋子、坐标和胜利路径绘制 |
| `ui/history_panel.py` | 右侧下棋记录和 AI 状态面板 |
| `game_mode.py` | 统一四种游戏模式及其人类玩家映射 |
| `game_timer.py` | 基于单调时钟的红蓝双方计时器 |
| `game_records.py` | JSON 保存、加载和逐步复盘 |
| `report_export.py` | ?????CSV?MCTS ??????? |
| `reports/README.md` | ???????? |
| `ai/search_result.py` | MCTS SearchResult ???????? |
| `records/README.md` | 游戏记录目录说明 |
| `requirements-ai.txt` | 可选 PyTorch/NumPy 依赖 |
| `checkpoints/README.md` | 模型权重提交和版本管理说明 |
| `test_ai.py` | AI 契约、回退和战术测试 |
| `test_mcts.py` | 编码、网络和 MCTS 测试（无 torch 时跳过） |
| `test_ai_worker.py` | AI 后台线程、异常和回退测试 |
| `test_ui.py` | 渲染几何、窗口布局和引擎集成测试 |
| `test_p1.py` | 模式、计时和 JSON 复盘测试 |
| `test_p2.py` | SearchResult???????????? |
| `test_evaluation.py` | Dijkstra ?????? |
| `test_tactics.py` | ??????? |
| `test_hybrid_ai.py` | HybridAI ???????? |
| `hex_match.py` | 自动对局：胜率统计、对局记录、失败棋局复现、耗时统计 |
| `hex_game_test.py` | 自测脚本（7 组） |
| `hex_engine.py` | 对局引擎：回合、判胜、两局换色、比分与历史记录 |
| `test_hex_engine.py` | 引擎状态机与两局制流程单元测试（11 项） |
| `test_hex_board.py` | 棋盘坐标、非法输入和兼容接口单元测试（9 项） |
| `check.bat` | 一键运行游戏自测和全部单元测试 |
| `test_hex_match.py` | 自动对局、记录和复盘回归测试（6 项） |
| `tools/benchmark_ai.py` | AI 固定种子、AB/BA 配对评测及 CSV/JSON 输出 |
| `test_benchmark_ai.py` | AI 评测工具回归测试（5 项） |
| `tools/check_model.py` | 模型文件、哈希、加载耗时和试下着法检查 |
| `docs/AI_INTERFACE.md` | AI 输入输出、模型加载、设备、耗时、种子和回退契约 |
| `docs/UI_ENGINE_CONTRACT.md` | 页面层与引擎层的接口和职责契约 |
| `hex_records.json` | 对局记录（自动生成，可删除） |
