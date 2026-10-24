# 模型文件目录

`NeuralMCTSAI` 默认读取：

```text
checkpoints/hex_model_v5.pt
```

## 当前参考模型

仓库当前附带的是一个 **集成验证参考模型**：

- 训练局数：10
- 训练种子：20261024
- checkpoint 类型：`reference_integration`
- 模型版本：`5.0.0`
- 编码版本：`3x11x11-v1`
- 推理设备：CPU
- 配置文件：`model_config.json`
- 校验文件：`hex_model_v5.pt.sha256`

该文件只用于验证模型加载、后台推理和 benchmark 流程，不代表最终比赛水平。正式提交前应由模型负责人使用更多训练局数生成新权重并替换。

## 验收命令

```powershell
python tools/check_model.py checkpoints/hex_model_v5.pt --time-limit-ms 200
```

## 正式模型要求

- 模型版本：`5.0.0`
- 编码版本：`3x11x11-v1`
- 记录 PyTorch 版本
- 记录训练局数和随机种子
- 记录文件 SHA256
- 无 GPU 环境也能完成 CPU 推理
