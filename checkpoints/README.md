# 模型文件目录

`NeuralMCTSAI` 默认读取：

```text
checkpoints/hex_model_v5.pt
```

## 当前正式模型

当前默认模型是从参考权重继续训练得到的正式模型：

- 训练目标：3000 局
- 训练种子：20261025
- 最优策略门禁：第 2000 局
- checkpoint 类型：`official`
- 模型版本：`5.0.0`
- 编码版本：`3x11x11-v1`
- 推理设备：CPU
- SHA256：`1918441ea6bd91c5be4bbd87bd09aa6be74937dca05226bee8e78c0f63d2c505`

策略网络单独对 BasicAI 的门禁胜率为 20%。完整神经网络 MCTS 在 200 局 AB/BA 配对评测中：

- 神经 AI 胜率：66.5%
- 95% 置信区间：59.7%～72.7%
- 神经 AI 执红胜率：72%
- 神经 AI 执蓝胜率：61%
- 平均手数：73.59
- 平均单步耗时：147.2 ms
- 最大单步耗时：389.5 ms
- 异常：0

详细数据见：

```text
checkpoints/training_summary.json
checkpoints/training_history.csv
benchmark_results/official_v5_200.json
```

## 其他权重

- `hex_model_v5_reference.pt`：最初训练 10 局的集成验证权重
- `hex_model_v5_best.pt`：训练过程门禁选出的最佳权重
- `hex_model_v5_trial*.pt`：100 局试训产物

## 验收命令

```powershell
python tools/check_model.py checkpoints/hex_model_v5.pt --time-limit-ms 200
python tools/benchmark_ai.py --ai-a basic --ai-b neural --games 200 --seed 20261025
```

## 正式模型要求

- 模型版本：`5.0.0`
- 编码版本：`3x11x11-v1`
- 记录 PyTorch 版本
- 记录训练局数和随机种子
- 记录文件 SHA256
- 无 GPU 环境也能完成 CPU 推理
