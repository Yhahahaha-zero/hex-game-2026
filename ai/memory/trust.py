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


def _sigmoid(value):
    clipped = max(-40.0, min(40.0, float(value)))
    return 1.0 / (1.0 + math.exp(-clipped))


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
    calibration_scale: float = 1.0
    calibration_bias: float = 0.0
    calibration_method: str = "identity"
    calibration_samples: int = 0

    @classmethod
    def untrained(cls):
        size = len(MemoryTrustFeatures(0, 0, 0, 0, False, 1, "exact", False).vector())
        return cls(weights=[0.0] * size, bias=0.0, metrics={})

    def raw_logit(self, features):
        vector = features.vector()
        if len(vector) != len(self.weights):
            raise ValueError("trust feature size mismatch")
        value = float(np.dot(vector, np.asarray(self.weights)) + self.bias)
        return max(-40.0, min(40.0, value))

    def predict_raw(self, features):
        return _sigmoid(self.raw_logit(features))

    def predict(self, features):
        raw_logit = self.raw_logit(features)
        calibrated_logit = (
            float(self.calibration_scale) * raw_logit
            + float(self.calibration_bias)
        )
        return _sigmoid(calibrated_logit)

    @classmethod
    def fit(cls, feature_rows, labels, *, epochs=400, learning_rate=0.08,
            sample_weights=None):
        if not feature_rows or len(feature_rows) != len(labels):
            raise ValueError("training features and labels are required")
        x = np.stack([row.vector() for row in feature_rows])
        y = np.asarray([1.0 if int(label) else 0.0 for label in labels])
        weights_sample = np.asarray(
            sample_weights if sample_weights is not None else [1.0] * len(y),
            dtype=np.float64,
        )
        if (len(weights_sample) != len(y)
                or not np.all(np.isfinite(weights_sample))
                or np.any(weights_sample <= 0)):
            raise ValueError("sample_weights must be finite, positive and match labels")
        weights_sample = weights_sample / weights_sample.mean()
        model = cls.untrained()
        weights = np.asarray(model.weights, dtype=np.float64)
        bias = 0.0
        for _ in range(int(epochs)):
            logits = np.clip(x @ weights + bias, -40.0, 40.0)
            predictions = 1.0 / (1.0 + np.exp(-logits))
            error = (predictions - y) * weights_sample
            weights -= learning_rate * (x.T @ error / len(y))
            bias -= learning_rate * float(error.mean())
        model.weights = weights.tolist()
        model.bias = bias
        probabilities = model.predict_rows(feature_rows)
        model.metrics = evaluate_trust(probabilities, labels)
        model.trained_at = datetime.now().astimezone().isoformat(timespec="seconds")
        return model

    def calibrate(self, feature_rows, labels, *, epochs=300, learning_rate=0.05,
                  sample_weights=None):
        """Fit Platt scaling on a disjoint calibration set."""
        if len(feature_rows) != len(labels):
            raise ValueError("calibration features and labels must have equal length")
        labels_array = np.asarray(
            [1.0 if int(value) else 0.0 for value in labels], dtype=np.float64
        )
        self.calibration_samples = int(len(labels_array))
        if len(labels_array) < 2 or len(set(labels_array.tolist())) < 2:
            self.calibration_scale = 1.0
            self.calibration_bias = 0.0
            self.calibration_method = "identity_insufficient_classes"
            return self
        raw_logits = np.asarray(
            [self.raw_logit(row) for row in feature_rows], dtype=np.float64
        )
        weights_sample = np.asarray(
            sample_weights if sample_weights is not None else [1.0] * len(labels_array),
            dtype=np.float64,
        )
        if (len(weights_sample) != len(labels_array)
                or not np.all(np.isfinite(weights_sample))
                or np.any(weights_sample <= 0)):
            raise ValueError("calibration sample_weights must be finite, positive and match labels")
        weights_sample = weights_sample / weights_sample.mean()
        scale = 1.0
        offset = 0.0
        for _ in range(int(epochs)):
            probabilities = 1.0 / (
                1.0 + np.exp(-np.clip(scale * raw_logits + offset, -40.0, 40.0))
            )
            error = (probabilities - labels_array) * weights_sample
            scale -= learning_rate * float(np.mean(error * raw_logits))
            offset -= learning_rate * float(np.mean(error))
            scale = max(0.05, min(10.0, scale))
            offset = max(-20.0, min(20.0, offset))
        self.calibration_scale = float(scale)
        self.calibration_bias = float(offset)
        self.calibration_method = "platt_scaling"
        return self

    def predict_rows(self, feature_rows):
        return [self.predict(row) for row in feature_rows]

    def predict_raw_rows(self, feature_rows):
        return [self.predict_raw(row) for row in feature_rows]

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
    auc = auc_score(probabilities, labels)
    return {
        "samples": len(probabilities),
        "brier": round(brier, 6),
        "ece": round(ece, 6),
        "auc": round(auc, 6) if auc is not None else None,
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
    weights = [float(value) for value in document["weights"]]
    if not weights or not all(math.isfinite(value) for value in weights):
        raise ValueError("trust weights must be finite")
    bias = float(document["bias"])
    calibration_scale = float(document.get("calibration_scale", 1.0))
    calibration_bias = float(document.get("calibration_bias", 0.0))
    if not all(math.isfinite(value) for value in (
        bias, calibration_scale, calibration_bias
    )):
        raise ValueError("trust parameters must be finite")
    if calibration_scale <= 0.0:
        raise ValueError("calibration_scale must be positive")
    calibration_samples = int(document.get("calibration_samples", 0))
    if calibration_samples < 0:
        raise ValueError("calibration_samples must be non-negative")
    return MemoryTrustModel(
        weights=weights,
        bias=bias,
        metrics=dict(document.get("metrics", {})),
        feature_version=TRUST_FEATURE_VERSION,
        trained_at=str(document.get("trained_at", "")),
        calibration_scale=calibration_scale,
        calibration_bias=calibration_bias,
        calibration_method=str(document.get("calibration_method", "identity")),
        calibration_samples=calibration_samples,
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()