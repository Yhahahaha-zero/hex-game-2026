# -*- coding: utf-8 -*-
"""Optional VLM refinement limited to coordinate-labelled cell crops."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ai.analysis_result import extract_json_object
from ui.board_renderer import cell_center
from vision.board_detector import normalize_standard_board
from vision.image_loader import load_image
from vision.types import VisionValidationError


VISION_PROMPT_VERSION = "hex-vision-cells-v2"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "llm_cache" / "vision"
MAX_CELLS_PER_REQUEST = 12


def _cache_key(image_sha256, model_name, uncertain_cells):
    payload = {
        "image_sha256": image_sha256,
        "model_name": str(model_name),
        "prompt_version": VISION_PROMPT_VERSION,
        "uncertain_cells": [list(move) for move in uncertain_cells],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_cache(cache_dir, key):
    path = Path(cache_dir) / f"{key}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _write_cache(cache_dir, key, value):
    try:
        directory = Path(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{key}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
    except OSError:
        pass


def _contact_sheet(board_image, coordinates, cache_dir, key):
    """Build a 3-column sheet of labelled, enlarged cell crops."""
    from PIL import Image, ImageDraw
    try:
        if board_image is None:
            raise ValueError
    except Exception:
        raise VisionValidationError("缺少标准化棋盘图像")
    tile_w, tile_h = 190, 205
    crop_pad = 25
    columns = 3
    rows = (len(coordinates) + columns - 1) // columns
    sheet = Image.new("RGB", (tile_w * columns, tile_h * rows), (245, 240, 229))
    draw = ImageDraw.Draw(sheet)
    for index, (q, r) in enumerate(coordinates):
        cx, cy = cell_center(q, r)
        crop = board_image.crop((
            max(0, int(cx) - crop_pad),
            max(0, int(cy) - crop_pad),
            min(board_image.width, int(cx) + crop_pad),
            min(board_image.height, int(cy) + crop_pad),
        )).resize((150, 150), Image.Resampling.LANCZOS)
        x = (index % columns) * tile_w + 20
        y = (index // columns) * tile_h + 10
        sheet.paste(crop, (x, y))
        draw.rectangle((x - 2, y - 2, x + 151, y + 151), outline=(0, 0, 0), width=2)
        draw.text((x, y + 160), f"q={q}, r={r}, cell={r + 1}{chr(65 + q)}",
                  fill=(40, 35, 30))
    path = Path(cache_dir) / f"_request_{key}.png"
    sheet.save(path, format="PNG")
    return path


def refine_uncertain_cells(image_path, image_sha256, cells, uncertain_cells, *,
                           client, model_name, cache_dir=None, board_image=None,
                           timeout=60, max_tokens=1000,
                           enable_thinking=False, cache_enabled=True):
    """Return updated cells and warnings; only uncertain coordinates can change."""
    target = tuple(tuple(move) for move in uncertain_cells)
    if not target:
        return tuple(cells), (), "", False
    warnings = []
    if len(target) > MAX_CELLS_PER_REQUEST:
        warnings.append(
            f"视觉复核每次最多处理 {MAX_CELLS_PER_REQUEST} 个不确定格子"
        )
        target = target[:MAX_CELLS_PER_REQUEST]
    key = _cache_key(image_sha256, model_name, target)
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = _read_cache(cache_dir, key)
    cache_hit = payload is not None
    request_path = None
    if payload is None:
        if board_image is None:
            info, original = load_image(image_path)
            board_image, _ = normalize_standard_board(original)
        request_path = _contact_sheet(board_image, target, cache_dir, key)
        prompt = (
            "图片是按行列排列的单格棋盘裁剪图，每个裁剪图下方标注了坐标。"
            "只判断每个裁剪图中央是空位、红棋还是蓝棋。"
            "必须严格返回 JSON："
            '{"schema_version":1,"cells":[{"q":0,"r":0,"color":"EMPTY",'
            '"confidence":0.0,"reason":"简要依据"}],"warnings":[]}。'
            "每个坐标必须返回一次，不得输出未列出的坐标，不得输出额外文字。"
            f"需要判断的坐标：{json.dumps([list(move) for move in target])}"
        )
        try:
            response = client.analyze_image(
                prompt,
                request_path,
                model=model_name,
                timeout=timeout,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
            try:
                payload = extract_json_object(response.text)
            except Exception as exc:
                raise VisionValidationError("视觉模型返回了无效 JSON") from exc
            if cache_enabled:
                _write_cache(cache_dir, key, payload)
        finally:
            if request_path is not None:
                try:
                    request_path.unlink(missing_ok=True)
                except OSError:
                    pass

    if not isinstance(payload, dict):
        raise VisionValidationError("视觉模型响应必须是 JSON 对象")
    raw_cells = payload.get("cells", [])
    if not isinstance(raw_cells, list):
        raise VisionValidationError("视觉模型 cells 必须是数组")
    allowed = set(target)
    updates = {}
    for item in raw_cells:
        if not isinstance(item, dict):
            warnings.append("忽略了格式错误的视觉复核结果")
            continue
        q, r = item.get("q"), item.get("r")
        move = (q, r)
        if move not in allowed:
            warnings.append("忽略了不在不确定集合中的视觉坐标")
            continue
        color = item.get("color")
        if color not in {"EMPTY", "RED", "BLUE"}:
            warnings.append("忽略了非法视觉颜色")
            continue
        confidence = item.get("confidence", 0.0)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            confidence = 0.0
        confidence = max(0.0, min(1.0, float(confidence)))
        updates[move] = (color, confidence)

    updated = []
    for cell in cells:
        update = updates.get((cell.q, cell.r))
        if update:
            updated.append(type(cell)(
                q=cell.q, r=cell.r, color=update[0],
                confidence=round(update[1], 4), source="vlm",
            ))
        else:
            updated.append(cell)
    return tuple(updated), tuple(warnings), model_name, cache_hit