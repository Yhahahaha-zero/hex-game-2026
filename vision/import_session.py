# -*- coding: utf-8 -*-
"""Read-only image import orchestration and snapshot export."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time

from hex_board import BLUE, RED, HexBoard
from vision.board_detector import normalize_standard_board
from vision.board_validator import validate_cells
from vision.image_loader import load_image
from vision.stone_classifier import classify_board
from vision.types import VisionBoardState, VisionError
from vision.vlm_board_reader import refine_uncertain_cells


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent / "artifacts" / "vision"


@dataclass(frozen=True)
class VisionImportSession:
    state: VisionBoardState
    board: HexBoard
    normalized_image: object

    def save_snapshot(self, output_dir=None, name=None):
        root = Path(output_dir or DEFAULT_OUTPUT_DIR)
        root.mkdir(parents=True, exist_ok=True)
        stem = name or f"vision_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        path = root / f"{stem}.json"
        path.write_text(
            json.dumps(self.state.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path


def import_standard_image(image_path, *, min_confidence=0.78,
                          client=None, model_name="qwen3-vl-plus",
                          use_vlm=False, cache_dir=None,
                          vlm_timeout=60, vlm_max_tokens=1000,
                          max_image_mb=10, cache_enabled=True):
    started = time.perf_counter()
    info, image = load_image(image_path, max_size_bytes=int(max_image_mb) * 1024 * 1024)
    normalized, region = normalize_standard_board(image)
    cells, uncertain, classify_ms = classify_board(normalized, min_confidence)
    warnings = []
    cache_hit = False
    actual_model = ""
    if use_vlm and uncertain and client is not None:
        try:
            cells, vlm_warnings, actual_model, cache_hit = refine_uncertain_cells(
                info.path,
                info.sha256,
                cells,
                uncertain,
                client=client,
                model_name=model_name,
                board_image=normalized,
                cache_dir=cache_dir,
                timeout=vlm_timeout,
                max_tokens=vlm_max_tokens,
                enable_thinking=False,
                cache_enabled=cache_enabled,
            )
            warnings.extend(vlm_warnings)
        except VisionError as exc:
            warnings.append(f"VLM 复核失败: {exc}")
    elif use_vlm and uncertain:
        warnings.append("未配置视觉模型，低置信度格子需要人工确认")

    uncertain = tuple(
        (cell.q, cell.r) for cell in cells
        if cell.color == "UNKNOWN" or cell.confidence < min_confidence
    )
    validation = validate_cells(cells)
    warnings.extend(validation.warnings)
    board = HexBoard()
    for cell in cells:
        if cell.color == "RED":
            board.place(cell.q, cell.r, RED)
        elif cell.color == "BLUE":
            board.place(cell.q, cell.r, BLUE)
    elapsed_ms = (time.perf_counter() - started) * 1000
    state = VisionBoardState(
        schema_version=1,
        image_path=str(info.path),
        image_sha256=info.sha256,
        image_width=info.width,
        image_height=info.height,
        board_width=11,
        cells=cells,
        uncertain_cells=uncertain,
        side_to_move=validation.side_to_move,
        winner_color=validation.winner_color,
        valid=validation.valid and not uncertain,
        warnings=tuple(warnings),
        model_name=actual_model,
        elapsed_ms=round(elapsed_ms, 3),
        metadata={
            "region_mode": region.mode,
            "classify_ms": classify_ms,
            "cache_hit": cache_hit,
            "vlm_used": bool(actual_model),
        },
    )
    return VisionImportSession(state=state, board=board, normalized_image=normalized)