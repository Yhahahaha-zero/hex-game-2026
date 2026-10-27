# -*- coding: utf-8 -*-
"""Image loading, normalization and integrity checks."""
from __future__ import annotations

import hashlib
from pathlib import Path

from vision.types import ImageInfo, VisionDependencyError, VisionImageError


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MIN_IMAGE_SIDE = 240


def _dependencies():
    try:
        from PIL import Image, ImageOps
        import numpy as np
    except ImportError as exc:
        raise VisionDependencyError(
            "视觉功能需要 Pillow 和 numpy，请安装 requirements-vision.txt"
        ) from exc
    return Image, ImageOps, np


def load_image(path, max_size_bytes=MAX_IMAGE_BYTES):
    """Load and EXIF-normalize a local image without modifying the source."""
    Image, ImageOps, np = _dependencies()
    image_path = Path(path).expanduser().resolve()
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise VisionImageError(f"不支持的图片格式: {image_path.suffix or '无扩展名'}")
    if not image_path.is_file():
        raise VisionImageError(f"图片不存在: {image_path}")
    try:
        size_bytes = image_path.stat().st_size
    except OSError as exc:
        raise VisionImageError("无法读取图片文件") from exc
    if size_bytes <= 0:
        raise VisionImageError("图片文件为空")
    if size_bytes > MAX_IMAGE_BYTES:
        raise VisionImageError("图片文件超过 10MB 限制")

    try:
        with Image.open(image_path) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except Exception as exc:
        raise VisionImageError("图片损坏或无法解析") from exc
    width, height = image.size
    if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
        raise VisionImageError("图片分辨率过低")

    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    info = ImageInfo(
        path=image_path,
        sha256=digest,
        width=width,
        height=height,
        size_bytes=size_bytes,
    )
    return info, image