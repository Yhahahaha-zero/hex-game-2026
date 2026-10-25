# -*- coding: utf-8 -*-
"""验证神经网络权重能否加载并完成一次合法落子。"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.neural_ai import NeuralMCTSAI, resolve_model_path
from hex_board import HexBoard, RED


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description="检查神经 AI 模型")
    parser.add_argument("model", nargs="?", type=Path, default=None)
    parser.add_argument("--time-limit-ms", type=int, default=200)
    args = parser.parse_args(argv)
    selected_metadata = {}
    if args.model is None:
        args.model, selected_metadata = resolve_model_path()
        print(f"selection={selected_metadata}")

    if not args.model.is_file():
        print(f"[ERROR] 模型不存在：{args.model}")
        return 2

    try:
        ai = NeuralMCTSAI(
            model_path=args.model,
            time_limit_ms=args.time_limit_ms,
        )
        started = time.perf_counter()
        move, reason = ai.choose_move(HexBoard(), RED)
        move_ms = (time.perf_counter() - started) * 1000
    except Exception as exc:
        print(f"[ERROR] 模型检查失败：{exc}")
        return 1

    print(f"name={ai.name}")
    print(f"version={ai.version}")
    print(f"device={ai.device}")
    print(f"model={ai.model_path}")
    print(f"sha256={sha256(args.model)}")
    print(f"load_ms={ai.load_ms:.3f}")
    print(f"move={move}")
    print(f"move_ms={move_ms:.3f}")
    print(f"reason={reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
