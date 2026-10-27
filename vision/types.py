# -*- coding: utf-8 -*-
"""Shared data structures and errors for optional vision import."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class VisionError(RuntimeError):
    """Base class for safe vision import errors."""


class VisionDependencyError(VisionError):
    """Raised when Pillow or NumPy is unavailable."""


class VisionImageError(VisionError):
    """Raised when an image cannot be loaded or normalized safely."""


class VisionValidationError(VisionError):
    """Raised when recognized board data violates Hex constraints."""


@dataclass(frozen=True)
class ImageInfo:
    path: Path
    sha256: str
    width: int
    height: int
    size_bytes: int


@dataclass(frozen=True)
class BoardRegion:
    left: int
    top: int
    width: int
    height: int
    mode: str


@dataclass(frozen=True)
class VisionCell:
    q: int
    r: int
    color: str
    confidence: float
    source: str = "color"

    def as_dict(self):
        return {
            "q": self.q,
            "r": self.r,
            "color": self.color,
            "confidence": round(float(self.confidence), 4),
            "source": self.source,
        }


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    side_to_move: str | None
    winner_color: str | None
    red_count: int
    blue_count: int
    warnings: tuple[str, ...] = ()

    def as_dict(self):
        return {
            "valid": self.valid,
            "side_to_move": self.side_to_move,
            "winner_color": self.winner_color,
            "red_count": self.red_count,
            "blue_count": self.blue_count,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class VisionBoardState:
    schema_version: int
    image_path: str
    image_sha256: str
    image_width: int
    image_height: int
    board_width: int
    cells: tuple[VisionCell, ...]
    uncertain_cells: tuple[tuple[int, int], ...]
    side_to_move: str | None
    winner_color: str | None
    valid: bool
    warnings: tuple[str, ...]
    model_name: str = ""
    elapsed_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def cell_map(self):
        return {(cell.q, cell.r): cell.color for cell in self.cells}

    def board_rows(self):
        mapping = self.cell_map()
        return [
            "".join(
                "R" if mapping.get((q, r)) == "RED"
                else "B" if mapping.get((q, r)) == "BLUE"
                else "." if mapping.get((q, r)) == "EMPTY"
                else "?"
                for q in range(self.board_width)
            )
            for r in range(self.board_width)
        ]

    def as_dict(self):
        return {
            "schema_version": self.schema_version,
            "image_path": self.image_path,
            "image_sha256": self.image_sha256,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "board_size": self.board_width,
            "cells": [cell.as_dict() for cell in self.cells],
            "board_rows": self.board_rows(),
            "uncertain_cells": [list(move) for move in self.uncertain_cells],
            "side_to_move": self.side_to_move,
            "winner_color": self.winner_color,
            "valid": self.valid,
            "warnings": list(self.warnings),
            "model_name": self.model_name,
            "elapsed_ms": self.elapsed_ms,
            "metadata": dict(self.metadata),
        }