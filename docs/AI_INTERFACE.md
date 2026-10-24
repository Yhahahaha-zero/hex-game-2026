# AI 接口契约

本文档约定海克斯棋项目所有 AI（基础 AI、随机 AI、训练模型 AI）的统一接口和运行边界。页面、自动对局和评测工具只依赖本契约，不依赖具体模型实现。

## 1. 统一接口

所有 AI 必须实现：

```python
def choose_move(self, board, player):
    """返回 ((q, r), reason)。"""
```

### 输入

- `board`：当前 `HexBoard` 对象；
- `player`：当前执棋颜色，只能是 `RED` 或 `BLUE`；
- 调用方负责保证 `player` 对应当前回合；
- `choose_move()` 只处理当前局面，不负责换手或判胜。

### 输出

必须返回二元组：

```python
((q, r), reason)
```

其中：

- `(q, r)` 是 0～10 内的整数棋盘坐标；
- 该坐标必须对应空位；
- `reason` 是字符串，用于界面显示、日志和复盘；
- 不得返回 `None`，不得只返回坐标，也不得返回其他结构。

示例：

```python
return (5, 5), "中心开局"
return (0, 10), "直接获胜"
return (7, 3), "模型评估 0.82"
```

### 异常

AI 无法产生合法着法时，应抛出带有明确信息的异常，例如：

```python
raise RuntimeError("模型推理失败：输入尺寸不匹配")
raise ValueError("模型未产生合法着法")
```

调用方不要在 AI 内部静默修改棋盘来“修正”错误。

## 2. 棋盘只读约束

AI 读取棋盘时必须遵守：

- 允许调用 `legal_moves()`、`candidate_moves()`、`is_empty()`、`would_win()`、`has_won()`、`winning_path()` 等读取或虚拟判断接口；
- 禁止调用 `place()` 修改正式棋盘；
- 禁止调用 `undo()` 修改正式历史；
- 禁止直接修改 `board.grid`、`board.history` 或内部缓存；
- 如果需要搜索，应复制局面或使用独立的虚拟棋盘。

推荐搜索方式：

```python
import copy

search_board = copy.deepcopy(board)
```

AI 只能返回着法，真正落子必须由页面或对局引擎调用 `HexEngine.play_move()` 完成。

## 3. 模型文件格式与路径

模型文件必须放在项目内或通过环境变量配置，禁止使用只存在于某位成员电脑上的绝对路径。

推荐目录：

```text
models/
├── hex_model_v1.0.0.npz
├── hex_model_v1.0.0.onnx
└── model_config_v1.0.0.json
```

支持格式由模型实现决定，建议优先使用：

- NumPy `.npz`：依赖少，适合轻量模型；
- ONNX `.onnx`：便于跨环境 CPU 推理；
- 小型 PyTorch `.pt`：必须确认提交电脑能够安装依赖。

模型配置建议包含：

```json
{
  "name": "HexModelAI",
  "version": "1.0.0",
  "format": "onnx",
  "input_shape": [11, 11, 3],
  "device": "cpu",
  "time_limit_ms": 200
}
```

模型类必须提供 `name` 属性。建议同时提供 `version`、`model_path` 和 `device` 属性，便于报告记录和问题追踪。

## 4. 模型加载方式

模型类必须支持无参数构造：

```python
class ModelAI:
    name = "模型AI"
    version = "1.0.0"

    def __init__(self):
        self.model = load_model(...)
        self.fallback = BasicAI()

    def choose_move(self, board, player):
        ...
```

### 可选随机种子接口

如果模型使用随机数，应实现：

```python
def set_seed(self, seed):
    ...
```

或：

```python
def seed(self, seed):
    ...
```

评测工具会在每局开始前调用其中一个方法。没有该接口时，评测工具仍会设置 Python 全局 `random.seed(seed)`。

### 可选重置接口

如果模型保存跨局状态，建议实现：

```python
def reset(self):
    ...
```

每次新对局开始前由模型自身清理临时状态。

### 模型构建要求

- 模型加载失败必须给出明确错误；
- 不得在 `choose_move()` 内部反复重新加载模型；
- 推理前必须切换到评估模式，例如 `model.eval()`；
- 必须使用无梯度推理，例如 `torch.no_grad()`；
- 不得依赖网络下载；
- 不得读取训练机的临时路径。

## 5. 单步时间限制

建议默认单步目标：

```text
目标：<= 200 ms
硬上限：<= 500 ms
```

实际限制以课程比赛通知为准。

模型必须自行控制搜索和推理时间：

- 固定深度搜索必须有最大深度；
- MCTS 必须设置时间或模拟次数上限；
- 模型首次加载时间不计入单步耗时，但必须单独记录；
- 单步超限不得卡死页面，页面后续应使用后台线程或超时保护。

评测工具会输出：

- `ai_a_average_move_ms`：AI A 平均单步耗时；
- `ai_a_max_move_ms`：AI A 最大单步耗时；
- AI B 对应统计。

## 6. CPU 和 GPU 要求

- 默认必须支持 CPU 推理；
- GPU 只能是性能增强，不能成为运行必需条件；
- 提交机的 Python、CUDA 和深度学习框架版本必须写入报告；
- 无法加载 GPU 时应自动回退 CPU，或在启动时给出明确错误；
- 最终验收必须在无 GPU 的干净环境中测试；
- 模型文件不应依赖训练机的绝对路径或环境变量。

## 7. 随机种子和可复现性

评测工具采用固定基础种子和 AB/BA 配对：

- 每对实验使用相同种子；
- 第一局 AI A 执红、AI B 执蓝；
- 第二局 AI A 执蓝、AI B 执红；
- 每轮实验使用独立种子；
- 所有随机选择应记录种子和模型版本。

AI 应尽量保证：

- 相同棋盘、相同颜色和相同种子产生相同着法；
- 使用 GPU 时固定随机种子并关闭非确定性算法；
- 不依赖系统时间、哈希随机顺序或未记录的随机源。

## 8. 模型版本管理

每次正式评测必须在模型类或评测命令中记录：

- 模型名称；
- 模型版本；
- 训练日期；
- 模型文件格式；
- 模型文件校验值；
- 推理设备；
- 单步时间限制；
- 使用的评测随机种子。

建议模型名称带版本，例如：

```python
name = "HexModelAI-v1.2.0"
```

模型文件更新后必须重新运行基准评测，不能沿用旧版本的胜率结论。

## 9. 模型加载失败与回退策略

### 正式游戏模式

页面运行时允许使用回退 AI：

```text
模型加载失败
    -> 记录错误
    -> 使用 BasicAI
    -> 界面显示“模型不可用，已回退基础 AI”
```

回退后必须在界面和日志中明确标识，不能看起来像模型正常工作。

### 严格评测模式

AI 对比评测默认不允许静默回退：

```text
模型异常
    -> 本局记为 error
    -> 记录异常类型和信息
    -> 继续运行后续局
    -> 最终报告异常率
```

否则模型经常失败但回退基础 AI 后仍会得到虚高胜率，实验结论不可信。

如果确实需要评测“带回退策略的模型”，应单独注明：

```text
ModelAI + BasicAI fallback
```

并与纯模型结果分开报告。

## 10. 推荐实现模板

```python
from hex_ai import BasicAI
from hex_board import RED, BLUE


class ModelAI:
    name = "HexModelAI-v1.0.0"
    version = "1.0.0"
    model_path = "models/hex_model_v1.0.0.onnx"
    device = "cpu"
    time_limit_ms = 200

    def __init__(self):
        self.model = self._load_model()
        self.fallback = BasicAI()

    def _load_model(self):
        # 在这里加载模型并执行版本检查
        return ...

    def set_seed(self, seed):
        # 设置模型、NumPy、torch 或第三方库的随机种子
        ...

    def reset(self):
        # 清理上一局的临时状态
        ...

    def choose_move(self, board, player):
        if player not in (RED, BLUE):
            raise ValueError(f"非法颜色: {player!r}")

        move = None
        reason = "模型推理"
        try:
            move = self._predict(board, player)
        except Exception as exc:
            raise RuntimeError(f"模型推理失败: {exc}") from exc

        if move is None:
            raise RuntimeError("模型没有返回着法")

        q, r = move
        if not board.in_range(q, r) or not board.is_empty(q, r):
            raise ValueError(f"模型返回非法着法: {(q, r)}")

        return (q, r), reason

    def _predict(self, board, player):
        # 将棋盘编码为模型输入并执行推理
        ...
```

## 11. 评测接入方式

内置 AI：

```bash
python tools/benchmark_ai.py --ai-a basic --ai-b random --games 500 --repeats 3
```

模块类：

```bash
python tools/benchmark_ai.py \
  --ai-a basic \
  --ai-b models.hex_model:ModelAI \
  --games 1000 \
  --repeats 3 \
  --seed 20261024
```

文件类：

```bash
python tools/benchmark_ai.py \
  --ai-a basic \
  --ai-b path/to/model_ai.py:ModelAI \
  --games 1000 \
  --repeats 3
```

## 12. 当前项目实现

- 模型入口：`ai.neural_ai.NeuralMCTSAI`；
- 页面回退工厂：`ai.neural_ai.create_ai_with_fallback()`；
- 默认权重：`checkpoints/hex_model_v5.pt`；
- 默认设备：有 CUDA 时使用 CUDA，否则使用 CPU；
- 默认单步上限：`200 ms`；
- 默认最大模拟次数：`100`；
- 页面通过后台线程加载和推理，不阻塞 pygame 主循环；`n- 神经 AI 可通过 `set_progress_callback()` 上报 MCTS 模拟进度；
- torch、权重或推理失败时回退 `BasicAI`，并在界面显示回退状态；
- 严格 benchmark 不回退，异常会写入结果；
- 可选依赖：`requirements-ai.txt`；
- 当前附带 `checkpoints/hex_model_v5.pt` 为训练 10 局的集成验证参考模型，正式比赛前应替换；
- 训练入口：`python ai/train_v5.py --games 500 --seed 20261024`。

### 神经 AI 评测命令

```bash
python tools/benchmark_ai.py --ai-a basic --ai-b neural --games 1000 --repeats 3 --seed 20261024
```

## 13. 验收清单


- `choose_move(board, player)` 返回 `((q, r), reason)`；
- 返回坐标始终在棋盘内且为空位；
- AI 不直接修改正式棋盘和历史；
- 模型只从项目相对路径或环境变量加载；
- 无 GPU 环境也能完成 CPU 推理；
- 支持固定随机种子；
- 首次加载时间和单步耗时可统计；
- 模型版本和文件可追溯；
- 严格评测不会静默回退基础 AI；
- 正式游戏回退时界面有明确提示；
- 通过 `check.bat` 和 AI 基准评测。
