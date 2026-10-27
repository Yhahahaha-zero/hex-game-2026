# -*- coding: utf-8 -*-
"""Import and validate a standard Hex screenshot, optionally with VLM help."""
import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.import_session import DEFAULT_OUTPUT_DIR, import_standard_image
from vision.types import VisionError


def main(argv=None):
    parser = argparse.ArgumentParser(description="标准海克斯棋盘截图识别")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--vlm", action="store_true",
                        help="让 qwen3-vl-plus 复核低置信度格子")
    parser.add_argument("--min-confidence", type=float, default=0.78)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--name", default=None)
    parser.add_argument("--json", action="store_true",
                        help="输出完整识别 JSON")
    args = parser.parse_args(argv)

    client = None
    model_name = ""
    if args.vlm:
        from ai.llm_client import create_llm_client
        from ai.llm_config import LLMConfigError, load_llm_config
        try:
            config = load_llm_config()
            if not config.enabled:
                raise RuntimeError("LLM_ENABLED=0，无法使用视觉模型复核")
            client = create_llm_client(config)
            model_name = config.vision_model
        except (LLMConfigError, RuntimeError) as exc:
            print(f"[ERROR] {exc}")
            return 2

    try:
        session = import_standard_image(
            args.image,
            min_confidence=args.min_confidence,
            client=client,
            model_name=model_name or "qwen3-vl-plus",
            use_vlm=args.vlm,
        )
        path = session.save_snapshot(args.output_dir, args.name)
    except VisionError as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 1
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 1

    state = session.state
    summary = {
        "valid": state.valid,
        "side_to_move": state.side_to_move,
        "winner_color": state.winner_color,
        "uncertain_cells": len(state.uncertain_cells),
        "warnings": len(state.warnings),
        "model_name": state.model_name,
        "elapsed_ms": state.elapsed_ms,
        "cache_hit": state.metadata.get("cache_hit", False),
        "output": str(path),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if args.json:
        print(json.dumps(state.as_dict(), ensure_ascii=False, indent=2))
    return 0 if state.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())