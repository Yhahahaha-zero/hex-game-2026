# -*- coding: utf-8 -*-
"""HexNetV5 自对弈训练、续训、固定评测和最佳模型保存。"""
import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
from hex_ai import BasicAI, RandomAI
from hex_board import BLUE, RED
from hex_engine import HexEngine
from hex_match import play_game

CHECKPOINT_DIR = PROJECT_ROOT.parent / "models"
EVALUATION_DIR = PROJECT_ROOT.parent / "evaluation" / "training"
DEFAULT_MODEL_PATH = CHECKPOINT_DIR / "hex_model_v5.pt"
DEFAULT_BEST_PATH = CHECKPOINT_DIR / "hex_model_v5_best.pt"
DEFAULT_LATEST_PATH = CHECKPOINT_DIR / "hex_model_v5_latest.pt"
DEFAULT_BEST_JSON = EVALUATION_DIR / "best.json"
DEFAULT_HISTORY_PATH = EVALUATION_DIR / "training_history.csv"


@dataclass(frozen=True)
class ReplaySample:
    state: object
    legal_mask: object
    action: int
    value_target: float


class ReplayBuffer:
    """?????????????????"""

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.samples = deque(maxlen=capacity)

    def __len__(self):
        return len(self.samples)

    def add_game(self, states, legal_masks, actions, value_targets):
        for state, mask, action, target in zip(
                states, legal_masks, actions, value_targets):
            self.samples.append(ReplaySample(state, mask, action, target))

    def sample(self, batch_size):
        count = min(batch_size, len(self.samples))
        if count == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        return random.sample(list(self.samples), count)


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
    states, players, legal_masks, actions = [], [], [], []
    while engine.state == "playing":
        board = engine.board
        player = engine.current_color
        states.append(encode_board(board, player))
        legal_masks.append(get_legal_mask(board, "cpu"))
        q, r = sample_move(model, board, player, device, temperature)
        players.append(player)
        actions.append(move_to_index(q, r))
        engine.play_move(q, r, reason="self-play")
    return states, players, legal_masks, actions, engine.winner_color


def create_value_targets(players, winner, device):
    targets = [1.0 if player == winner else -1.0 for player in players]
    return torch.tensor(targets, dtype=torch.float32, device=device)


def train_one_game(model, optimizer, device, temperature, replay, batch_size):
    states, players, legal_masks, actions, winner = play_self_game(
        model, device, temperature
    )
    value_targets = [1.0 if player == winner else -1.0 for player in players]
    replay.add_game(states, legal_masks, actions, value_targets)
    batch = replay.sample(batch_size)

    batch_states = torch.stack([item.state for item in batch]).to(device)
    batch_masks = torch.stack([item.legal_mask for item in batch]).to(device)
    batch_actions = torch.tensor(
        [item.action for item in batch], dtype=torch.long, device=device
    )
    batch_targets = torch.tensor(
        [item.value_target for item in batch], dtype=torch.float32, device=device
    )

    policy_logits, values = model(batch_states)
    masked_logits = policy_logits.masked_fill(~batch_masks, float("-inf"))
    log_probs = F.log_softmax(masked_logits, dim=1)
    selected_log_probs = log_probs[
        torch.arange(len(batch_actions), device=device), batch_actions
    ]
    policy_loss = -(selected_log_probs * batch_targets.detach()).mean()
    value_loss = F.mse_loss(values, batch_targets)
    loss = policy_loss + 0.5 * value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item(), policy_loss.item(), value_loss.item(), len(states), winner


class PolicyAI:
    """训练门禁用的快速策略网络 AI，不运行 MCTS。"""
    name = "PolicyEvalAI"

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def choose_move(self, board, player):
        state = encode_board(board, player).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(state)
        mask = get_legal_mask(board, self.device)
        logits = logits[0].masked_fill(~mask, float("-inf"))
        index = int(torch.argmax(logits).item())
        return (index % 11, index // 11), "policy-gate"


def evaluate_policy(model, device, games, seed, opponent_name):
    if games <= 0 or games % 2:
        raise ValueError("评测局数必须是大于0的偶数")
    opponent_type = BasicAI if opponent_name == "basic" else RandomAI
    policy_ai = PolicyAI(model, device)
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    model.eval()
    wins = losses = 0
    red_wins = red_games = blue_wins = blue_games = 0
    total_moves = 0
    total_model_ms = 0.0
    model_move_count = 0
    max_model_ms = 0.0
    try:
        for index in range(games):
            random.seed(seed + index)
            opponent = opponent_type()
            if index % 2 == 0:
                record = play_game(policy_ai, opponent)
                red_games += 1
                total_moves += record["moves_count"]
                for move in record["moves"]:
                    if move["p"] == "红":
                        total_model_ms += move["ms"]
                        model_move_count += 1
                        max_model_ms = max(max_model_ms, move["ms"])
                if record["winner"] == "红":
                    wins += 1
                    red_wins += 1
                else:
                    losses += 1
            else:
                record = play_game(opponent, policy_ai)
                blue_games += 1
                total_moves += record["moves_count"]
                for move in record["moves"]:
                    if move["p"] == "蓝":
                        total_model_ms += move["ms"]
                        model_move_count += 1
                        max_model_ms = max(max_model_ms, move["ms"])
                if record["winner"] == "蓝":
                    wins += 1
                    blue_wins += 1
                else:
                    losses += 1
    finally:
        model.train()
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    return {
        "games": games, "wins": wins, "losses": losses,
        "win_rate": wins / games,
        "red_wins": red_wins, "red_games": red_games,
        "blue_wins": blue_wins, "blue_games": blue_games,
        "average_moves": total_moves / games,
        "average_move_ms": total_model_ms / max(1, model_move_count),
        "max_move_ms": max_model_ms,
    }


def write_model_config(path, digest, metadata):
    path = Path(path)
    document = {
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "model_version": MODEL_VERSION,
        "encoding_version": ENCODING_VERSION,
        "model_file": path.name,
        "sha256": digest,
        **metadata,
    }
    path.with_name(f"{path.stem}_config.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if path.name == DEFAULT_MODEL_PATH.name:
        (path.parent / "model_config.json").write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )


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
    write_model_config(path, digest, metadata)
    return digest


def promote_checkpoint(source, target, metadata):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(target.suffix + ".sha256").write_text(
        f"{digest}  {target.name}\n", encoding="ascii"
    )
    write_model_config(target, digest, metadata)


def append_history(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp", "completed_games", "session_game", "eval_games",
        "wins", "losses", "win_rate", "red_win_rate", "blue_win_rate",
        "average_moves", "average_move_ms", "max_move_ms",
        "best_win_rate", "is_best", "loss", "policy_loss", "value_loss",
    ]
    new_file = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def load_resume(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        completed_games = int(checkpoint.get("games", 0))
    else:
        state_dict = checkpoint
        completed_games = 0
    model.load_state_dict(state_dict)
    return completed_games


def make_metadata(args, completed, device, started, metrics=None, digest=None):
    metadata = {
        "model_version": MODEL_VERSION,
        "training_games": completed,
        "games": completed,
        "checkpoint_kind": args.checkpoint_kind,
        "seed": args.seed,
        "opponent": args.eval_opponent,
        "created_at": started.isoformat(timespec="seconds"),
        "trained_at": started.isoformat(timespec="seconds"),
        "pytorch_version": str(torch.__version__),
        "device": device,
        "temperature": args.temperature,
    }
    if metrics is not None:
        metadata.update({
            "win_rate": metrics["win_rate"],
            "eval_win_rate": metrics["win_rate"],
            "eval_games": metrics["games"],
            "average_moves": metrics["average_moves"],
            "average_move_ms": metrics["average_move_ms"],
            "max_move_ms": metrics["max_move_ms"],
        })
    if digest:
        metadata["sha256"] = digest
    return metadata


def write_best_json(path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train(args):
    if args.games <= 0 or args.save_interval <= 0:
        raise ValueError("games ? save_interval ????0")
    if args.eval_games <= 0 or args.eval_games % 2:
        raise ValueError("eval_games ?????0???")
    if args.replay_size <= 0 or args.batch_size <= 0:
        raise ValueError("replay_size ? batch_size ????0")
    set_training_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HexNetV5().to(device)
    completed_before = 0
    if args.resume:
        completed_before = load_resume(model, args.resume, device)
        print(f"? {args.resume} ?????? {completed_before} ?")
    if args.games <= completed_before:
        raise ValueError("--games ???? checkpoint ???????")
    model.train()
    optimizer = Adam(model.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.replay_size)
    red_wins = blue_wins = total_moves = 0
    total_loss = total_policy = total_value = 0.0
    started = datetime.now()
    best_rate = -1.0
    best_metadata = None
    latest_metadata = None
    session_games = args.games - completed_before

    if args.eval_every > 0:
        metrics = evaluate_policy(
            model, device, args.eval_games, args.eval_seed, args.eval_opponent
        )
        best_rate = metrics["win_rate"]
        digest = save_checkpoint(
            model, args.best_output,
            make_metadata(args, completed_before, device, started, metrics),
        )
        best_metadata = make_metadata(
            args, completed_before, device, started, metrics, digest
        )
        write_best_json(args.best_json, best_metadata)
        append_history(args.history_output, {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "completed_games": completed_before, "session_game": 0,
            "eval_games": args.eval_games, "wins": metrics["wins"],
            "losses": metrics["losses"], "win_rate": round(best_rate, 6),
            "red_win_rate": metrics["red_wins"] / max(1, metrics["red_games"]),
            "blue_win_rate": metrics["blue_wins"] / max(1, metrics["blue_games"]),
            "average_moves": round(metrics["average_moves"], 3),
            "average_move_ms": round(metrics["average_move_ms"], 3),
            "max_move_ms": round(metrics["max_move_ms"], 3),
            "best_win_rate": round(best_rate, 6), "is_best": 1,
            "loss": 0, "policy_loss": 0, "value_loss": 0,
        })
        print(f"?????? {args.eval_opponent} ?? {best_rate:.2%}")

    for session_game in range(1, session_games + 1):
        loss, policy_loss, value_loss, moves, winner = train_one_game(
            model, optimizer, device, args.temperature, replay, args.batch_size
        )
        completed = completed_before + session_game
        total_loss += loss
        total_policy += policy_loss
        total_value += value_loss
        total_moves += moves
        red_wins += winner == RED
        blue_wins += winner == BLUE

        if completed % args.save_interval == 0 or session_game == session_games:
            latest_metadata = make_metadata(
                args, completed, device, started, digest=None
            )
            latest_metadata.update({
                "red_wins": red_wins,
                "blue_wins": blue_wins,
                "win_rate": best_rate if best_rate >= 0 else None,
            })
            digest = save_checkpoint(model, args.latest_output, latest_metadata)
            latest_metadata["sha256"] = digest

        evaluated = args.eval_every > 0 and completed % args.eval_every == 0
        if evaluated:
            metrics = evaluate_policy(
                model, device, args.eval_games, args.eval_seed, args.eval_opponent
            )
            is_best = metrics["win_rate"] > best_rate
            if is_best:
                best_rate = metrics["win_rate"]
                digest = save_checkpoint(
                    model, args.best_output,
                    make_metadata(args, completed, device, started, metrics),
                )
                best_metadata = make_metadata(
                    args, completed, device, started, metrics, digest
                )
                write_best_json(args.best_json, best_metadata)
            append_history(args.history_output, {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "completed_games": completed, "session_game": session_game,
                "eval_games": args.eval_games, "wins": metrics["wins"],
                "losses": metrics["losses"], "win_rate": round(metrics["win_rate"], 6),
                "red_win_rate": metrics["red_wins"] / max(1, metrics["red_games"]),
                "blue_win_rate": metrics["blue_wins"] / max(1, metrics["blue_games"]),
                "average_moves": round(metrics["average_moves"], 3),
                "average_move_ms": round(metrics["average_move_ms"], 3),
                "max_move_ms": round(metrics["max_move_ms"], 3),
                "best_win_rate": round(best_rate, 6), "is_best": int(is_best),
                "loss": round(total_loss / session_game, 6),
                "policy_loss": round(total_policy / session_game, 6),
                "value_loss": round(total_value / session_game, 6),
            })
            print(f"?? {completed} ???? {metrics['win_rate']:.2%} | ?? {best_rate:.2%}")
            if completed % args.save_interval == 0 or session_game == session_games:
                latest_metadata = make_metadata(
                    args, completed, device, started, metrics
                )
                latest_metadata.update({"red_wins": red_wins, "blue_wins": blue_wins})
                digest = save_checkpoint(model, args.latest_output, latest_metadata)
                latest_metadata["sha256"] = digest

        if session_game % 10 == 0:
            print(
                f"? {completed:04d}/{args.games} ? | "
                f"RED {red_wins} | BLUE {blue_wins} | "
                f"Loss {total_loss / session_game:.4f} | "
                f"???? {total_moves / session_game:.1f} | "
                f"Replay {len(replay)}"
            )

    if best_metadata is None:
        if latest_metadata is None:
            latest_metadata = make_metadata(args, args.games, device, started)
            digest = save_checkpoint(model, args.latest_output, latest_metadata)
            latest_metadata["sha256"] = digest
        best_metadata = dict(latest_metadata)
        write_best_json(args.best_json, best_metadata)
        promote_checkpoint(args.latest_output, args.best_output, best_metadata)
    promote_checkpoint(args.best_output, args.output, best_metadata)
    print(f"??????????????{args.output}")
    print(f"????????{args.latest_output}")
    print(f"???????{best_rate:.2%}")
    return best_rate


def main(argv=None):
    parser = argparse.ArgumentParser(description="训练 HexNetV5")
    parser.add_argument("--games", type=int, default=500, help="累计目标训练局数")
    parser.add_argument("--resume", type=Path, default=None, help="从 checkpoint 续训")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--eval-seed", type=int, default=20262025)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-opponent", choices=("basic", "random"), default="basic")
    parser.add_argument("--checkpoint-kind", default="trained")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--latest-output", type=Path, default=DEFAULT_LATEST_PATH)
    parser.add_argument("--best-output", type=Path, default=DEFAULT_BEST_PATH)
    parser.add_argument("--best-json", type=Path, default=DEFAULT_BEST_JSON)
    parser.add_argument("--history-output", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--replay-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    try:
        train(args)
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
