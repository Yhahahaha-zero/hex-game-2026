import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from hex_board import ROWS, COLS, RED, BLUE
from hex_engine import HexEngine
from ai.encode import encode_board


# ============================================================
# V5 配置
# ======================
# =====================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SELF_PLAY_GAMES = 500
LEARNING_RATE = 0.0003
VALUE_WEIGHT = 0.5
TEMPERATURE = 1.2

SAVE_INTERVAL = 50

MODEL_PATH = "checkpoints/hex_model_v5.pt"


# ============================================================
# V5 Policy + Value 网络
# ============================================================

class HexNetV5(nn.Module):

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
            nn.ReLU()
        )

        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                64 * ROWS * COLS,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                ROWS * COLS
            )
        )

        # Value Head
        self.value_head = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                64 * ROWS * COLS,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                1
            ),

            nn.Tanh()
        )

    def forward(self, x):

        features = self.features(x)

        policy_logits = self.policy_head(features)

        value = self.value_head(features)

        value = value.squeeze(1)

        return policy_logits, value


# ============================================================
# 工具函数
# ============================================================

def move_to_index(q, r):
    return r * COLS + q


def index_to_move(index):
    r = index // COLS
    q = index % COLS
    return q, r


def get_legal_moves(board):
    moves = []

    for r in range(ROWS):
        for q in range(COLS):

            if board.grid[r][q] == 0:
                moves.append((q, r))

    return moves


def get_legal_mask(board, device):

    mask = torch.zeros(
        ROWS * COLS,
        dtype=torch.bool,
        device=device
    )

    for q, r in get_legal_moves(board):

        index = move_to_index(q, r)

        mask[index] = True

    return mask


# ============================================================
# 神经网络选择落子
# ============================================================

def sample_move(
    model,
    board,
    player,
    device,
    temperature=1.0
):

    state = encode_board(
        board,
        player
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        policy_logits, value = model(state)

        policy_logits = policy_logits[0]

    legal_mask = get_legal_mask(
        board,
        device
    )

    masked_logits = torch.full_like(
        policy_logits,
        float("-inf")
    )

    masked_logits[legal_mask] = policy_logits[legal_mask]

    temperature = max(
        temperature,
        0.01
    )

    probabilities = torch.softmax(
        masked_logits / temperature,
        dim=0
    )

    distribution = torch.distributions.Categorical(
        probabilities
    )

    index = distribution.sample().item()

    move = index_to_move(index)

    probability = probabilities[index].item()

    return move, probability


# ============================================================
# 自我对弈
# ============================================================

def play_self_game(
    model,
    device,
    temperature
):

    engine = HexEngine()

    board = engine.board

    states = []
    players = []
    legal_masks = []
    actions = []

    player = RED

    while True:

        state = encode_board(
            board,
            player
        )

        legal_mask = get_legal_mask(
            board,
            device="cpu"
        )

        move, probability = sample_move(
            model,
            board,
            player,
            device,
            temperature
        )

        q, r = move

        states.append(state)

        players.append(player)

        legal_masks.append(
            legal_mask
        )

        actions.append(
            move_to_index(q, r)
        )

        engine.play(
            q,
            r,
            reason="neural"
        )

        if engine.winner_color is not None:

            winner = engine.winner_color

            break

        if player == RED:
            player = BLUE
        else:
            player = RED

    return (
        states,
        players,
        legal_masks,
        actions,
        winner
    )


# ============================================================
# Value 训练目标
# ============================================================

def create_value_targets(
    players,
    winner,
    device
):

    targets = []

    for player in players:

        if player == winner:
            targets.append(1.0)
        else:
            targets.append(-1.0)

    return torch.tensor(
        targets,
        dtype=torch.float32,
        device=device
    )


# ============================================================
# 单局训练
# ============================================================

def train_one_game(
    model,
    optimizer
):

    (
        states,
        players,
        legal_masks,
        actions,
        winner
    ) = play_self_game(
        model,
        DEVICE,
        TEMPERATURE
    )

    states = torch.stack(
        states
    ).to(DEVICE)

    legal_masks = torch.stack(
        legal_masks
    ).to(DEVICE)

    actions = torch.tensor(
        actions,
        dtype=torch.long,
        device=DEVICE
    )

    value_targets = create_value_targets(
        players,
        winner,
        DEVICE
    )

    # 前向传播
    policy_logits, values = model(
        states
    )

    # Policy Loss
    masked_logits = policy_logits.masked_fill(
        ~legal_masks,
        float("-inf")
    )

    log_probs = F.log_softmax(
        masked_logits,
        dim=1
    )

    selected_log_probs = log_probs[
        torch.arange(
            len(actions),
            device=DEVICE
        ),
        actions
    ]

    advantages = value_targets.detach()

    policy_loss = -(
        selected_log_probs * advantages
    ).mean()

    # Value Loss
    value_loss = F.mse_loss(
        values,
        value_targets
    )

    # 总 Loss
    loss = (
        policy_loss
        + VALUE_WEIGHT * value_loss
    )

    # 反向传播
    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    return (
        loss.item(),
        policy_loss.item(),
        value_loss.item(),
        len(actions),
        winner
    )


# ============================================================
# 主训练函数
# ============================================================

def train():

    print("=" * 60)
    print("Hex V5 Policy + Value 训练")
    print("=" * 60)

    print("设备:", DEVICE)
    print("训练局数:", SELF_PLAY_GAMES)
    print("学习率:", LEARNING_RATE)
    print("温度:", TEMPERATURE)

    print()

    model = HexNetV5().to(DEVICE)

    model.train()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    red_wins = 0
    blue_wins = 0

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_moves = 0

    for game in range(
        1,
        SELF_PLAY_GAMES + 1
    ):

        (
            loss,
            policy_loss,
            value_loss,
            moves,
            winner
        ) = train_one_game(
            model,
            optimizer
        )

        total_loss += loss
        total_policy_loss += policy_loss
        total_value_loss += value_loss
        total_moves += moves

        if winner == RED:
            red_wins += 1
        else:
            blue_wins += 1

        # 定期保存
        if (
            game % SAVE_INTERVAL == 0
            or game == SELF_PLAY_GAMES
        ):

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "game":
                        game,

                    "red_wins":
                        red_wins,

                    "blue_wins":
                        blue_wins
                },
                MODEL_PATH
            )

        # 输出进度
        if game % 10 == 0:

            avg_loss = (
                total_loss / game
            )

            avg_moves = (
                total_moves / game
            )

            print(
                f"第 {game:03d}/{SELF_PLAY_GAMES} 局 | "
                f"RED {red_wins} | "
                f"BLUE {blue_wins} | "
                f"Loss {avg_loss:.4f} | "
                f"平均步数 {avg_moves:.1f}"
            )

    print()
    print("=" * 60)
    print("V5 训练完成！")
    print("=" * 60)

    print(
        "总训练局数:",
        SELF_PLAY_GAMES
    )

    print(
        "RED 胜局:",
        red_wins
    )

    print(
        "BLUE 胜局:",
        blue_wins
    )

    print(
        f"RED 胜率: "
        f"{red_wins / SELF_PLAY_GAMES * 100:.1f}%"
    )

    print(
        f"BLUE 胜率: "
        f"{blue_wins / SELF_PLAY_GAMES * 100:.1f}%"
    )

    print(
        f"平均 Loss: "
        f"{total_loss / SELF_PLAY_GAMES:.6f}"
    )

    print(
        f"平均 Policy Loss: "
        f"{total_policy_loss / SELF_PLAY_GAMES:.6f}"
    )

    print(
        f"平均 Value Loss: "
        f"{total_value_loss / SELF_PLAY_GAMES:.6f}"
    )

    print(
        f"平均对局步数: "
        f"{total_moves / SELF_PLAY_GAMES:.1f}"
    )

    print()

    print(
        "V5 模型已保存到:"
    )

    print(
        MODEL_PATH
    )

    print("=" * 60)


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    train()