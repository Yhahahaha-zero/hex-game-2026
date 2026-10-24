# -*- coding: utf-8 -*-

"""
神经网络 Hex AI

当前采用已经测试过的最佳版本：

    V5 神经网络
        +
    V7 MCTS

模型：
    checkpoints/hex_model_v5.pt

MCTS：
    100 次搜索

接口与原来的 BasicAI 保持一致：

    choose_move(board, player)

返回：

    ((q, r), "MCTS")
"""

import os

import torch

from ai.model_v5 import HexNetV5
from ai.mcts import MCTS


class NeuralMCTSAI:
    """
    V7 神经网络 + MCTS AI。
    """

    name = "神经MCTS AI"

    def __init__(
        self,
        model_path="checkpoints/hex_model_v5.pt",
        device=None,
        simulations=100,
        c_puct=1.5
    ):

        # ----------------------------------------------------
        # 设备
        # ----------------------------------------------------

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device

        # ----------------------------------------------------
        # 检查模型文件
        # ----------------------------------------------------

        if not os.path.isfile(model_path):

            raise FileNotFoundError(
                f"找不到神经网络模型：{model_path}"
            )

        self.model_path = model_path

        # ----------------------------------------------------
        # 创建模型
        # ----------------------------------------------------

        self.model = HexNetV5().to(
            self.device
        )

        # ----------------------------------------------------
        # 加载 V5 checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        # V5 保存的是 checkpoint 字典
        if isinstance(checkpoint, dict) and \
                "model_state_dict" in checkpoint:

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        else:

            # 兼容直接保存 state_dict 的情况
            self.model.load_state_dict(
                checkpoint
            )

        self.model.eval()

        # ----------------------------------------------------
        # MCTS
        # ----------------------------------------------------

        self.mcts = MCTS(
            model=self.model,
            device=self.device,
            simulations=simulations,
            c_puct=c_puct
        )

    # ========================================================
    # 选择落子
    # ========================================================

    def choose_move(
        self,
        board,
        player
    ):

        # V7 MCTS.search() 返回：
        # best_move, best_visits, root

        move, visits, root = self.mcts.search(
            board,
            player
        )

        return move, f"MCTS {visits}次搜索"
