# -*- coding: utf-8 -*-
"""Versioned retrieval policy manifest."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile


POLICY_SCHEMA_VERSION = 1
VALID_RETRIEVAL_TYPES = ("exact", "canonical", "similar")


@dataclass(frozen=True)
class RetrievalPolicy:
    top_k: int = 5
    minimum_similarity: float = 0.72
    minimum_support: int = 3
    minimum_confidence: float = 0.55
    alpha: float = 0.15
    retrieval_types: tuple[str, ...] = VALID_RETRIEVAL_TYPES
    schema_version: int = POLICY_SCHEMA_VERSION
    created_at: str | None = None
    benchmark_file: str | None = None

    def as_dict(self):
        data = asdict(self)
        data["retrieval_types"] = list(self.retrieval_types)
        return data


def load_retrieval_policy(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(document.get("schema_version", 0)) != POLICY_SCHEMA_VERSION:
        raise ValueError("unsupported retrieval policy schema")
    types = tuple(document.get("retrieval_types", VALID_RETRIEVAL_TYPES))
    if not types or any(item not in VALID_RETRIEVAL_TYPES for item in types):
        raise ValueError("invalid retrieval_types")
    policy = RetrievalPolicy(
        top_k=max(1, int(document.get("top_k", 5))),
        minimum_similarity=max(0.0, min(1.0, float(
            document.get("minimum_similarity", 0.72)
        ))),
        minimum_support=max(1, int(document.get("minimum_support", 3))),
        minimum_confidence=max(0.0, min(1.0, float(
            document.get("minimum_confidence", 0.55)
        ))),
        alpha=max(0.0, min(1.0, float(document.get("alpha", 0.15)))),
        retrieval_types=types,
        created_at=document.get("created_at"),
        benchmark_file=document.get("benchmark_file"),
    )
    return policy


def save_retrieval_policy(path, policy):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = policy.as_dict()
    if not document.get("created_at"):
        document["created_at"] = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if Path(temp_name).exists():
            Path(temp_name).unlink()
    return path