# AI 算法融合说明

本版本以 `hex-game-2026-main` 为主体，已将前两个压缩包中的核心 AI 思路移植为可直接适配本项目 `HexBoard` 的实现。

## 已融合

### 1. AI-for-Hex
- Dijkstra / 最短连接路径评估
- 从双方最短路径提取关键候选点
- 浅层 Minimax 风格的最强回应预览
- 必胜/必防战术搜索

### 2. hexai-ai
- MCTS（Monte Carlo Tree Search）
- UCT 选择思想
- 多次快速模拟
- 启发式引导 rollout，而不是纯随机 rollout

### 3. 原第三项目
- 保持 `HexBoard`、`HexEngine`、pygame 界面和两局制规则不变
- `BasicAI.choose_move(board, player)` 接口不变
- 因此原 GUI 不需要重新编写
- `random` AI 仍保留，可用于对比测试

## AI 决策顺序

1. 空棋盘中心开局
2. 一步必胜
3. 一步必防
4. Dijkstra 双方路径筛选
5. 浅层 Minimax 风格回应评估
6. MCTS 短时搜索
7. 路径评估兜底

默认 MCTS 思考时间约 0.32 秒/步。性能较低的电脑可把 `HybridAI(time_limit=0.32)` 改为 `0.15`～`0.20`；性能较高可提高到 `0.5` 左右。

## 运行

```bash
py hex_game.py
```

选择：
- 2：玩家执红，AI 执蓝
- 3：AI 执红，玩家执蓝
- 4：AI vs AI 观战

自动测试：

```bash
py hex_game_test.py
```

AI 对 AI：

```bash
py hex_match.py --games 100 --ai-red hybrid --ai-blue hybrid
```

随机 AI 基准：

```bash
py hex_match.py --games 100 --ai-red hybrid --ai-blue random
```

## 注意

本版本没有依赖 OpenAI API，因此不需要 API Key，也不会触发你之前遇到的 `401 Unauthorized`。这里的“AI”是本地算法 AI：Dijkstra + Minimax 风格搜索 + MCTS。
