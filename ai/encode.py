import torch

from hex_board import ROWS, COLS, EMPTY, RED, BLUE


def encode_board(board, player):
    """
    将 Hex 棋盘转换为神经网络可以处理的张量。

    输入：
        board  : HexBoard 对象
        player : 当前玩家（RED 或 BLUE）

    输出：
        shape = [3, 11, 11]

        第 0 层：当前玩家的棋子
        第 1 层：对手的棋子
        第 2 层：当前玩家身份
                   RED -> 全 1
                   BLUE -> 全 0
    """

    state = torch.zeros(
        (3, ROWS, COLS),
        dtype=torch.float32
    )

    opponent = BLUE if player == RED else RED

    for r in range(ROWS):
        for q in range(COLS):
            value = board.grid[r][q]

            if value == player:
                state[0, r, q] = 1.0

            elif value == opponent:
                state[1, r, q] = 1.0

    # 玩家身份信息
    if player == RED:
        state[2, :, :] = 1.0

    return state
