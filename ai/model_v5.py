# -*- coding: utf-8 -*-
"""HexNetV5 policy/value 网络定义和动作空间工具。"""
import torch
from torch import nn

from hex_board import COLS, EMPTY, ROWS


MODEL_NAME = "HexNetV5"
MODEL_VERSION = "5.0.0"
ENCODING_VERSION = "3x11x11-v1"
POLICY_SIZE = ROWS * COLS


class HexNetV5(nn.Module):
    """输入 ``[B, 3, 11, 11]``，输出 policy logits 和 value。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * ROWS * COLS, 256),
            nn.ReLU(),
            nn.Linear(256, POLICY_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * ROWS * COLS, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        features = self.features(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features).squeeze(1)
        return policy_logits, value


def move_to_index(q, r):
    """将内部 ``(q, r)`` 坐标转换为 0..120 的动作编号。"""
    if not (0 <= q < COLS and 0 <= r < ROWS):
        raise ValueError(f"非法坐标: {(q, r)!r}")
    return r * COLS + q


def index_to_move(index):
    """将 0..120 的动作编号转换为内部 ``(q, r)`` 坐标。"""
    if not (0 <= index < POLICY_SIZE):
        raise ValueError(f"非法动作编号: {index!r}")
    return index % COLS, index // COLS


def get_legal_moves(board):
    """返回全部空位，动作顺序与策略输出编号一致。"""
    return [
        (q, r)
        for r in range(ROWS)
        for q in range(COLS)
        if board.grid[r][q] == EMPTY
    ]


def get_legal_mask(board, device):
    """返回形状为 ``[121]`` 的布尔合法动作掩码。"""
    mask = torch.zeros(POLICY_SIZE, dtype=torch.bool, device=device)
    for q, r in get_legal_moves(board):
        mask[move_to_index(q, r)] = True
    return mask
