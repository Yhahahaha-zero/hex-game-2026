# -*- coding: utf-8 -*-
"""Apply promotion gates to a V2 memory-trained model candidate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.memory.model_registry import (
    PromotionGate,
    promote_candidate,
    rollback_selected_model,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate")
    parser.add_argument("--benchmark")
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT.parent / "models"))
    parser.add_argument("--selected-manifest")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rollback", action="store_true")
    parser.add_argument(
        "--minimum-games", type=int,
        default=int(os.environ.get("HEX_MEMORY_PROMOTION_MIN_GAMES", "1000")),
    )
    parser.add_argument(
        "--minimum-win-rate", type=float,
        default=float(os.environ.get("HEX_MEMORY_PROMOTION_MIN_WIN_RATE", "50")),
    )
    parser.add_argument(
        "--minimum-color-win-rate", type=float,
        default=float(os.environ.get("HEX_MEMORY_PROMOTION_MIN_COLOR_WIN_RATE", "40")),
    )
    parser.add_argument("--maximum-errors", type=int, default=0)
    parser.add_argument(
        "--maximum-move-ms", type=float,
        default=float(os.environ.get("HEX_MEMORY_PROMOTION_MAX_MOVE_MS", "500")),
    )
    args = parser.parse_args(argv)
    if args.rollback:
        manifest = rollback_selected_model(
            models_dir=args.models_dir,
            selected_manifest_path=args.selected_manifest,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if not args.candidate or not args.benchmark:
        raise ValueError("--candidate and --benchmark are required without --rollback")
    report = promote_candidate(
        candidate_path=args.candidate,
        benchmark_path=args.benchmark,
        models_dir=args.models_dir,
        selected_manifest_path=args.selected_manifest,
        gate=PromotionGate(
            minimum_games=args.minimum_games,
            minimum_win_rate=args.minimum_win_rate,
            minimum_color_win_rate=args.minimum_color_win_rate,
            maximum_errors=args.maximum_errors,
            maximum_move_ms=args.maximum_move_ms,
        ),
        apply=args.apply,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())