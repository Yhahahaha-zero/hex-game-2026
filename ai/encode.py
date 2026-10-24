# -*- coding: utf-8 -*-
"""将 Hex 棋盘编码为神经网络输入。"""
import torch

from hex_board import BLUE, COLS, RED, ROWS


ENCODING_VERSION = "3x11x11-v1"


def encode_board(board, player):
    """返回当前玩家视角的 ``[3, 11, 11]`` 浮点张量。"""
    if player not in (RED, BLUE):
        raise ValueError(f"非法颜色编号: {player!r}")

    state = torch.zeros((3, ROWS, COLS), dtype=torch.float32)
    opponent = BLUE if player == RED else RED

    for row in range(ROWS):
        for col in range(COLS):
            value = board.grid[row][col]
            if value == player:
                state[0, row, col] = 1.0
            elif value == opponent:
                state[1, row, col] = 1.0

    if player == RED:
        state[2, :, :] = 1.0
    return state
