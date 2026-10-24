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
py -m unittest -v test_hex_board.py test_hex_engine.py test_hex_match.py
```

## 文件说明

| 文件 | 职责 |
|---|---|
| `hex_game.py` | pygame 图形界面（主程序） |
| `hex_board.py` | 规则核心：棋盘、BFS/DFS 连通性判定、并查集快速判定、文本棋盘 |
| `hex_ai.py` | AI：直接获胜 > 紧急防守 > 中心偏好随机；预留路径评估扩展点 |
| `hex_match.py` | 自动对局：胜率统计、对局记录、失败棋局复现、耗时统计 |
| `hex_game_test.py` | 自测脚本（7 组） |
| `hex_engine.py` | 对局引擎：回合、判胜、两局换色、比分与历史记录 |
| `test_hex_engine.py` | 引擎状态机与两局制流程单元测试（11 项） |
| `test_hex_board.py` | 棋盘坐标、非法输入和兼容接口单元测试（9 项） |
| `check.bat` | 一键运行游戏自测和全部单元测试 |
| `test_hex_match.py` | 自动对局、记录和复盘回归测试（6 项） |
| `docs/UI_ENGINE_CONTRACT.md` | 页面层与引擎层的接口和职责契约 |
| `hex_records.json` | 对局记录（自动生成，可删除） |
