# -*- coding: utf-8 -*-
"""Data, model and memory-effectiveness drift snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import uuid

from ai.memory.store import DriftSnapshotRecord


@dataclass(frozen=True)
class DriftThresholds:
    minimum_validation_rate: float = 0.90
    minimum_mean_reward: float = -0.10
    maximum_negative_rate: float = 0.65
    maximum_retrieval_failure_rate: float = 0.25
    maximum_validation_drop: float = 0.10
    maximum_reward_drop: float = 0.20


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_drift_snapshot(store, *, thresholds=None, sample_limit=1000):
    thresholds = thresholds or DriftThresholds()
    decisions = store.list_recent_decisions(limit=sample_limit)
    outcomes = store.list_recent_retrieval_outcomes(limit=sample_limit)
    validated = sum(
        1 for row in decisions if row["decision_quality"] == "validated"
    )
    validation_rate = validated / len(decisions) if decisions else 0.0
    applied = [row for row in outcomes if int(row["applied"])]
    rewards = [float(row["retrieval_reward"]) for row in applied]
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    negative_rate = (
        sum(1 for value in rewards if value < 0) / len(rewards)
        if rewards else 0.0
    )
    retrieval_failure_rate = (
        sum(1 for row in outcomes if row["retrieval_mode"] == "shadow"
            and row["similarity"] is None) / len(outcomes)
        if outcomes else 0.0
    )
    previous_rows = store.list_drift_snapshots(limit=1)
    previous = None
    if previous_rows:
        try:
            previous = json.loads(previous_rows[0]["metrics_json"])
        except (TypeError, ValueError):
            previous = None
    baseline_delta = {}
    if previous:
        baseline_delta = {
            "validation_rate": round(
                validation_rate - float(previous.get("validation_rate", 0.0)), 6
            ),
            "mean_reward": round(
                mean_reward - float(previous.get("mean_reward", 0.0)), 6
            ),
        }
    alerts = []
    if decisions and validation_rate < thresholds.minimum_validation_rate:
        alerts.append("validation_rate_low")
    if applied and mean_reward < thresholds.minimum_mean_reward:
        alerts.append("memory_reward_low")
    if applied and negative_rate > thresholds.maximum_negative_rate:
        alerts.append("memory_negative_rate_high")
    if outcomes and retrieval_failure_rate > thresholds.maximum_retrieval_failure_rate:
        alerts.append("retrieval_failure_rate_high")
    if (previous and baseline_delta["validation_rate"]
            < -thresholds.maximum_validation_drop):
        alerts.append("validation_rate_drop")
    if (previous and baseline_delta["mean_reward"]
            < -thresholds.maximum_reward_drop):
        alerts.append("memory_reward_drop")
    metrics = {
        "decisions": len(decisions),
        "validated": validated,
        "validation_rate": round(validation_rate, 6),
        "outcomes": len(outcomes),
        "applied": len(applied),
        "mean_reward": round(mean_reward, 6),
        "negative_rate": round(negative_rate, 6),
        "retrieval_failure_rate": round(retrieval_failure_rate, 6),
        "baseline_delta": baseline_delta,
    }
    now = _now()
    timestamps = [
        str(row["created_at"]) for row in decisions + outcomes
        if row.get("created_at")
    ]
    record = DriftSnapshotRecord(
        snapshot_id=str(uuid.uuid4()),
        window_start=min(timestamps) if timestamps else now,
        window_end=max(timestamps) if timestamps else now,
        metrics_json=json.dumps(metrics, ensure_ascii=False, sort_keys=True),
        alerts_json=json.dumps(alerts, ensure_ascii=False),
        created_at=now,
    )
    store.save_drift_snapshot(record)
    return {"metrics": metrics, "alerts": alerts, "snapshot_id": record.snapshot_id}