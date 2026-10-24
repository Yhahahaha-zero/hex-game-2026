# -*- coding: utf-8 -*-
"""HexNetV5 自对弈训练入口。"""
import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from ai.encode import encode_board
from ai.model_v5 import (ENCODING_VERSION, MODEL_NAME, MODEL_VERSION,
                         HexNetV5, get_legal_mask, move_to_index)
from hex_board import BLUE, RED
from hex_engine import HexEngine

DEFAULT_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "hex_model_v5.pt"


def set_training_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError):
        pass


def sample_move(model, board, player, device, temperature):
    state = encode_board(board, player).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, _ = model(state)
        policy_logits = policy_logits[0]

    legal_mask = get_legal_mask(board, device)
    masked_logits = torch.full_like(policy_logits, float("-inf"))
    masked_logits[legal_mask] = policy_logits[legal_mask]
    probabilities = torch.softmax(masked_logits / max(temperature, 0.01), dim=0)
    index = torch.distributions.Categorical(probabilities).sample().item()
    return index % 11, index // 11


def play_self_game(model, device, temperature):
    engine = HexEngine()
    states = []
    players = []
    legal_masks = []
    actions = []

    while engine.state == "playing":
        board = engine.board
        player = engine.current_color
        state = encode_board(board, player)
        legal_masks.append(get_legal_mask(board, "cpu"))
        q, r = sample_move(model, board, player, device, temperature)
        states.append(state)
        players.append(player)
        actions.append(move_to_index(q, r))
        engine.play_move(q, r, reason="self-play")

    return (
        states,
        players,
        legal_masks,
        actions,
        engine.winner_color,
    )


def create_value_targets(players, winner, device):
    targets = [1.0 if player == winner else -1.0 for player in players]
    return torch.tensor(targets, dtype=torch.float32, device=device)


def train_one_game(model, optimizer, device, temperature):
    states, players, legal_masks, actions, winner = play_self_game(
        model, device, temperature
    )
    states = torch.stack(states).to(device)
    legal_masks = torch.stack(legal_masks).to(device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    value_targets = create_value_targets(players, winner, device)

    policy_logits, values = model(states)
    masked_logits = policy_logits.masked_fill(~legal_masks, float("-inf"))
    log_probs = F.log_softmax(masked_logits, dim=1)
    selected_log_probs = log_probs[
        torch.arange(len(actions), device=device), actions
    ]
    policy_loss = -(selected_log_probs * value_targets.detach()).mean()
    value_loss = F.mse_loss(values, value_targets)
    loss = policy_loss + 0.5 * value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item(), policy_loss.item(), value_loss.item(), len(actions), winner


def save_checkpoint(model, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "encoding_version": ENCODING_VERSION,
        **metadata,
    }
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    config_path = path.with_name("model_config.json")
    config_path.write_text(
        json.dumps(
            {
                "name": MODEL_NAME,
                "version": MODEL_VERSION,
                "encoding_version": ENCODING_VERSION,
                "model_file": path.name,
                "sha256": digest,
                **metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def train(args):
    set_training_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HexNetV5().to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr=args.lr)
    red_wins = blue_wins = total_moves = 0
    total_loss = total_policy = total_value = 0.0
    started = datetime.now()

    for game in range(1, args.games + 1):
        loss, policy_loss, value_loss, moves, winner = train_one_game(
            model, optimizer, device, args.temperature
        )
        total_loss += loss
        total_policy += policy_loss
        total_value += value_loss
        total_moves += moves
        red_wins += winner == RED
        blue_wins += winner == BLUE

        if game % args.save_interval == 0 or game == args.games:
            save_checkpoint(
                model,
                args.output,
                {
                    "trained_at": started.isoformat(timespec="seconds"),
                    "games": game,
                    "checkpoint_kind": args.checkpoint_kind,
                    "seed": args.seed,
                    "temperature": args.temperature,
                    "device": device,
                    "pytorch_version": str(torch.__version__),
                    "red_wins": red_wins,
                    "blue_wins": blue_wins,
                },
            )

        if game % 10 == 0:
            print(
                f"第 {game:03d}/{args.games} 局 | RED {red_wins} | BLUE {blue_wins} | "
                f"Loss {total_loss / game:.4f} | 平均步数 {total_moves / game:.1f}"
            )

    print(f"训练完成，模型保存到：{args.output}")
    print(f"RED {red_wins} / BLUE {blue_wins}")
    print(f"平均 Loss {total_loss / args.games:.6f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="训练 HexNetV5")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--checkpoint-kind", default="trained")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args(argv)
    if args.games <= 0 or args.save_interval <= 0:
        parser.error("games 和 save-interval 必须大于 0")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

