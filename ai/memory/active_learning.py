# -*- coding: utf-8 -*-
"""Active-learning prioritization for high-value memory positions."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json


@dataclass(frozen=True)
class ActiveLearningScore:
    candidate_id: str
    position_id: str
    decision_id: str
    priority: float
    reasons: tuple[str, ...]


def score_candidate(*, position_id, decision_id, trust_probability,
                    confidence, support_count, conflict=False,
                    retrieval_type="similar", ply=1, minimum_support=3):
    reasons = []
    uncertainty = 1.0 - abs(float(trust_probability) - 0.5) * 2.0
    reasons.append(f"trust_uncertainty:{uncertainty:.4f}")
    support_gap = max(0.0, 1.0 - abs(int(support_count) - int(minimum_support)) / 5.0)
    if support_gap > 0.5:
        reasons.append("support_near_threshold")
    conflict_bonus = 0.25 if conflict else 0.0
    if conflict:
        reasons.append("memory_conflict")
    type_bonus = {"exact": 0.0, "canonical": 0.15, "similar": 0.30}.get(
        retrieval_type, 0.0
    )
    if retrieval_type == "similar":
        reasons.append("approximate_retrieval")
    late_bonus = min(0.15, max(0, int(ply) - 20) / 200.0)
    priority = (
        0.35 * uncertainty
        + 0.20 * support_gap
        + 0.15 * max(0.0, min(1.0, float(confidence)))
        + conflict_bonus
        + type_bonus
        + late_bonus
    )
    identity = f"active1:{position_id}:{decision_id}".encode("utf-8")
    return ActiveLearningScore(
        candidate_id="act1-" + hashlib.sha256(identity).hexdigest()[:32],
        position_id=str(position_id),
        decision_id=str(decision_id),
        priority=round(min(1.0, priority), 6),
        reasons=tuple(reasons),
    )


def select_candidates(scores, *, quota):
    if int(quota) < 1:
        raise ValueError("quota must be positive")
    ordered = sorted(
        scores,
        key=lambda item: (-item.priority, item.position_id, item.decision_id),
    )
    return tuple(ordered[:int(quota)])


def persist_candidates(store, scores, *, dataset_id=None):
    from ai.memory.store import ActiveLearningCandidateRecord
    created = 0
    for score in scores:
        record = ActiveLearningCandidateRecord(
            candidate_id=score.candidate_id,
            position_id=score.position_id,
            decision_id=score.decision_id,
            priority=score.priority,
            reason_json=json.dumps(
                list(score.reasons), ensure_ascii=False
            ),
            selected=1,
            dataset_id=dataset_id,
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        created += int(store.save_active_learning_candidate(record))
    return created