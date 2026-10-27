# -*- coding: utf-8 -*-
"""Train a calibrated V4 memory trust model from retrieval outcomes."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.memory.dataset import split_match_ids
from ai.memory.schema import memory_db_path
from ai.memory.store import ExperienceStore, TrustModelRecord
from ai.memory.trust import (
    MemoryTrustFeatures,
    MemoryTrustModel,
    evaluate_trust,
    file_sha256,
    save_trust_model,
)


def _features(event, outcome):
    evidence = json.loads(event.get("evidence_json") or "{}")
    hits = evidence.get("hits", [])
    top = hits[0] if hits else {}
    return MemoryTrustFeatures(
        similarity=float(outcome.get("similarity") or 0.0),
        confidence=float(outcome.get("confidence") or 0.0),
        support_count=int(evidence.get("support_count", 0) or 0),
        vote_share=float(evidence.get("vote_share", 0.0) or 0.0),
        conflict=bool(evidence.get("conflict", False)),
        ply=int(event.get("query_ply") or 1),
        retrieval_type=str(top.get("similarity_type", "similar")),
        model_agrees=bool(event.get("base_q") == event.get("memory_q")
                          and event.get("base_r") == event.get("memory_r")),
    )


def _selection_weight(event):
    """Return IPW metadata while treating legacy NULL probabilities as observed."""
    raw = event.get("selection_probability")
    if raw is None:
        return 1.0, "legacy_probability_missing"
    try:
        probability = float(raw)
    except (TypeError, ValueError):
        return None, "invalid_selection_probability"
    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        return None, "invalid_selection_probability"
    if probability == 0.0:
        return None, "zero_selection_probability"
    weight = min(10.0, 1.0 / max(0.1, probability))
    return weight, "observed_probability"


def _grouped_splits(samples, *, seed):
    split = split_match_ids(
        [sample[0] for sample in samples],
        seed=seed,
        ratios=(0.60, 0.20, 0.20),
    )
    fit_match_ids = list(split["train"])
    calibration_match_ids = []
    if len(fit_match_ids) > 1:
        ordered = sorted(
            fit_match_ids,
            key=lambda value: hashlib.sha256(
                f"{seed}:calibration:{value}".encode("utf-8")
            ).hexdigest(),
        )
        calibration_count = max(1, int(round(len(ordered) * 0.25)))
        calibration_count = min(calibration_count, len(ordered) - 1)
        calibration_match_ids = ordered[:calibration_count]
        fit_set = set(calibration_match_ids)
        fit_match_ids = [value for value in fit_match_ids if value not in fit_set]
    assignment = {}
    for name, values in (
        ("train", fit_match_ids),
        ("calibration", calibration_match_ids),
        ("validation", split["validation"]),
        ("test", split["test"]),
    ):
        for match_id in values:
            assignment[match_id] = name
    grouped = {"train": [], "calibration": [], "validation": [], "test": []}
    for sample in samples:
        grouped[assignment.get(sample[0], "train")].append(sample)
    return grouped



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(memory_db_path()))
    parser.add_argument("--output", default=str(
        PROJECT_ROOT.parent / "models" / "memory_trust_model.json"
    ))
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--minimum-samples", type=int, default=20)
    parser.add_argument("--minimum-per-class", type=int, default=5)
    args = parser.parse_args(argv)
    with ExperienceStore(args.db) as store:
        outcomes = [
            row for row in store.list_retrieval_outcomes(limit=1000000)
            if int(row["applied"])
        ]
        events = {
            row["retrieval_event_id"]: row
            for row in store.list_retrieval_events(limit=1000000)
        }
        samples = []
        excluded = []
        weight_sources = {"legacy_probability_missing": 0, "observed_probability": 0}
        for outcome in outcomes:
            event = events.get(outcome["retrieval_event_id"])
            if event is None:
                excluded.append({
                    "retrieval_event_id": outcome["retrieval_event_id"],
                    "reason": "retrieval_event_missing",
                })
                continue
            weight, weight_source = _selection_weight(event)
            if weight is None:
                excluded.append({
                    "retrieval_event_id": outcome["retrieval_event_id"],
                    "reason": weight_source,
                })
                continue
            weight_sources[weight_source] += 1
            samples.append((
                str(event.get("query_match_id") or "unknown"),
                _features(event, outcome),
                1 if float(outcome["retrieval_reward"]) > 0 else 0,
                weight,
            ))
        if len(samples) < args.minimum_samples:
            raise SystemExit(
                f"Need at least {args.minimum_samples} applied samples, found {len(samples)}"
            )
        positives = sum(sample[2] for sample in samples)
        negatives = len(samples) - positives
        if min(positives, negatives) < args.minimum_per_class:
            raise SystemExit(
                "Both positive and negative trust samples are required: "
                f"positive={positives}, negative={negatives}"
            )
        grouped = _grouped_splits(samples, seed=20261025)
        train = grouped["train"] or samples
        calibration = grouped["calibration"]
        validation = grouped["validation"] or train
        test = grouped["test"] or validation
        train_positives = sum(sample[2] for sample in train)
        train_negatives = len(train) - train_positives
        if (min(train_positives, train_negatives) < args.minimum_per_class
                and args.minimum_per_class > 0):
            raise SystemExit(
                "The training split must contain both classes after match grouping: "
                f"positive={train_positives}, negative={train_negatives}"
            )
        model = MemoryTrustModel.fit(
            [item[1] for item in train],
            [item[2] for item in train],
            sample_weights=[item[3] for item in train],
            epochs=args.epochs,
        )

        raw_validation = evaluate_trust(
            model.predict_raw_rows([item[1] for item in validation]),
            [item[2] for item in validation],
        )
        raw_test = evaluate_trust(
            model.predict_raw_rows([item[1] for item in test]),
            [item[2] for item in test],
        )
        if calibration:
            model.calibrate(
                [item[1] for item in calibration],
                [item[2] for item in calibration],
                sample_weights=[item[3] for item in calibration],
            )
        else:
            model.calibration_method = "identity_no_calibration_set"
        fallbacks = []
        if not grouped["validation"]:
            fallbacks.append("validation_uses_train")
        if not grouped["test"]:
            fallbacks.append("test_uses_validation")
        weights = [item[3] for item in train]
        model.metrics = {
            "train": evaluate_trust(
                model.predict_rows([item[1] for item in train]),
                [item[2] for item in train],
            ),
            "calibration": evaluate_trust(
                model.predict_rows([item[1] for item in calibration]),
                [item[2] for item in calibration],
            ),
            "validation": evaluate_trust(
                model.predict_rows([item[1] for item in validation]),
                [item[2] for item in validation],
            ),
            "test": evaluate_trust(
                model.predict_rows([item[1] for item in test]),
                [item[2] for item in test],
            ),
            "raw_validation": raw_validation,
            "raw_test": raw_test,
            "split_counts": {
                name: len(values) for name, values in grouped.items()
            },
            "evaluation_fallbacks": fallbacks,
            "calibration_method": model.calibration_method,
            "calibration_scale": round(model.calibration_scale, 6),
            "calibration_bias": round(model.calibration_bias, 6),
            "calibration_samples": model.calibration_samples,
            "selection_probability_sources": weight_sources,
            "excluded_outcomes": excluded,
            "effective_sample_size": round(
                sum(weights) ** 2 / max(1e-12, sum(value ** 2 for value in weights)),
                3,
            ),
        }
        path = save_trust_model(args.output, model)
        model_id = str(uuid.uuid4())
        store.save_trust_model(TrustModelRecord(
            trust_model_id=model_id,
            model_path=str(Path(path).resolve()),
            sha256=file_sha256(path),
            feature_version=model.feature_version,
            metrics_json=json.dumps(model.metrics, sort_keys=True),
            status="candidate",
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        ))
    print(json.dumps({
        "trust_model_id": model_id,
        "path": str(path),
        "metrics": model.metrics,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())