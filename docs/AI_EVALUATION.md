# AI 对照评测与默认选择

## 结构兼容性

- `ai/evaluation.py`：Dijkstra 连通距离、最短路径、候选点评分；
- `ai/tactics.py`：两空桥、桥补点和双威胁；
- `ai/hybrid_ai.py`：路径评估 + 浅层预览 + 轻量 MCTS；
- `ai/ai_factory.py`：默认比赛 AI 工厂；
- `ai/neural_ai.py`：神经网络 + MCTS，支持 Dijkstra 先验。

## 评测结果

以下均为 AB/BA 交换红蓝、固定随机种子 `20261025`：

| 对局 | 局数 | 结果 |
|---|---:|---|
| HybridAI vs BasicAI | 100 | 86 : 14，HybridAI 86% |
| 加入距离先验的 NeuralMCTS vs BasicAI | 50 | 30 : 20，NeuralMCTS 60% |
| HybridAI vs NeuralMCTS | 50 | 36 : 14，HybridAI 72% |

HybridAI 执红胜率 92%，执蓝胜率 52%，最大单步约 414 ms。

## 默认选择

当前默认比赛 AI：

```text
HybridAI
```

原因：

- 对 BasicAI 的胜率明显高于当前 NeuralMCTS；
- 对 NeuralMCTS 的直接对抗胜率为 72%；
- 不依赖 PyTorch 和模型文件；
- 单步最大耗时低于 500 ms。

如需切换回神经模型：

```powershell
$env:HEX_AI_ENGINE = "neural"
```

可选值：

```text
hybrid
neural
basic
```

## 注意

- NeuralMCTS 仍然保留，可继续训练和评测；
- Dijkstra、桥和双威胁模块都只读取正式棋盘；
- 正式棋盘修改只由 HexEngine 完成。
