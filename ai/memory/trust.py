# -*- coding: utf-8 -*-
"""Explainable calibrated trust model for memory recommendations."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


TRUST_FEATURE_VERSION = 1
RETRIEVAL_TYPES = ("exact", "canonical", "similar")


@dataclass(frozen=True)
class MemoryTrustFeatures:
    similarity: float
    confidence: float
    support_count: int
    vote_share: float
    conflict: bool
    ply: int
    retrieval_type: str
    model_agrees: bool

    def vector(self):
        if self.retrieval_type not in RETRIEVAL_TYPES:
            raise ValueError("invalid retrieval_type")
        type_values = [1.0 if self.retrieval_type == value else 0.0 for value in RETRIEVAL_TYPES]
        return np.asarray([
            max(0.0, min(1.0, float(self.similarity))),
            max(0.0, min(1.0, float(self.confidence))),
            min(1.0, max(0.0, float(self.support_count) / 10.0)),
            max(0.0, min(1.0, float(self.vote_share))),
            0.0 if self.conflict else 1.0,
            max(0.0, min(1.0, float(self.ply) / 121.0)),
            *type_values,
            1.0 if self.model_agrees else 0.0,
        ], dtype=np.float64)


@dataclass
class MemoryTrustModel:
    weights: list[float]
    bias: float
    metrics: dict
    feature_version: int = TRUST_FEATURE_VERSION
    trained_at: str = ""

    @classmethod
    def untrained(cls):
        size = len(MemoryTrustFeatures(0, 0, 0, 0, False, 1, "exact", False).vector())
        return cls(weights=[0.0] * size, bias=0.0, metrics={})

    def predict(self, features):
        vector = features.vector()
        if len(vector) != len(self.weights):
            raise ValueError("trust feature size mismatch")
        value = float(np.dot(vector, np.asarray(self.weights)) + self.bias)
        value = max(-40.0, min(40.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    @classmethod
    def fit(cls, feature_rows, labels, *, epochs=400, learning_rate=0.08):
        if not feature_rows or len(feature_rows) != len(labels):
            raise ValueError("training features and labels are required")
        x = np.stack([row.vector() for row in feature_rows])
        y = np.asarray([1.0 if int(label) else 0.0 for label in labels])
        model = cls.untrained()
        weights = np.asarray(model.weights, dtype=np.float64)
        bias = 0.0
        for _ in range(int(epochs)):
            logits = np.clip(x @ weights + bias, -40.0, 40.0)
            predictions = 1.0 / (1.0 + np.exp(-logits))
            error = predictions - y
            weights -= learning_rate * (x.T @ error / len(y))
            bias -= learning_rate * float(error.mean())
        model.weights = weights.tolist()
        model.bias = bias
        probabilities = model.predict_rows(feature_rows)
        model.metrics = evaluate_trust(probabilities, labels)
        model.trained_at = datetime.now().astimezone().isoformat(timespec="seconds")
        return model

    def predict_rows(self, feature_rows):
        return [self.predict(row) for row in feature_rows]

    def as_dict(self):
        return asdict(self)


def auc_score(probabilities, labels):
    positives = [p for p, y in zip(probabilities, labels) if int(y)]
    negatives = [p for p, y in zip(probabilities, labels) if not int(y)]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def evaluate_trust(probabilities, labels):
    if not probabilities:
        return {}
    y = np.asarray([1.0 if int(value) else 0.0 for value in labels])
    p = np.asarray(probabilities, dtype=np.float64)
    brier = float(np.mean((p - y) ** 2))
    bins = [[] for _ in range(10)]
    for probability, label in zip(p, y):
        index = min(9, int(float(probability) * 10))
        bins[index].append((float(probability), float(label)))
    ece = 0.0
    for values in bins:
        if not values:
            continue
        avg_p = sum(item[0] for item in values) / len(values)
        avg_y = sum(item[1] for item in values) / len(values)
        ece += abs(avg_p - avg_y) * len(values) / len(values)
    return {
        "samples": len(probabilities),
        "brier": round(brier, 6),
        "ece": round(ece, 6),
        "auc": round(auc_score(probabilities, labels), 6)
        if auc_score(probabilities, labels) is not None else None,
        "positive_rate": round(float(y.mean()), 6),
    }


def save_trust_model(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(model.as_dict(), ensure_ascii=False, indent=2) + "\n"
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if Path(temp_name).exists():
            Path(temp_name).unlink()
    return path


def load_trust_model(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(document.get("feature_version", 0)) != TRUST_FEATURE_VERSION:
        raise ValueError("unsupported trust feature version")
    return MemoryTrustModel(
        weights=[float(value) for value in document["weights"]],
        bias=float(document["bias"]),
        metrics=dict(document.get("metrics", {})),
        feature_version=TRUST_FEATURE_VERSION,
        trained_at=str(document.get("trained_at", "")),
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()