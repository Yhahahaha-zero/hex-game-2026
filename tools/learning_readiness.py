# -*- coding: utf-8 -*-
"""Audit whether experience data is ready for training and promotion."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

TOOLS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.memory.schema import memory_db_path
from ai.memory.store import ExperienceStore


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_readiness_report(store, *, dataset_dir, candidate_path,
                           selected_model_path, candidate_gate_path=None,
                           trust_model_path=None, minimum_applied=20,
                           minimum_per_class=5):
    if minimum_applied < 1 or minimum_per_class < 1:
        raise ValueError("minimum sample thresholds must be positive")
    counts = store.counts()
    matches = store.list_matches(limit=1000000)
    completed_matches = sum(1 for row in matches if row.get("status") == "completed")
    outcomes = store.list_retrieval_outcomes(limit=1000000)
    applied = [row for row in outcomes if int(row.get("applied") or 0)]
    positive = sum(1 for row in applied if float(row.get("retrieval_reward") or 0.0) > 0)
    negative = sum(1 for row in applied if float(row.get("retrieval_reward") or 0.0) < 0)

    dataset_dir = Path(dataset_dir)
    manifest = _read_json(dataset_dir / "manifest.json")
    sample_count = int((manifest or {}).get("sample_count", 0) or 0)
    split_counts = dict((manifest or {}).get("split_counts", {}) or {})
    source_match_ids = list((manifest or {}).get("match_ids", []) or [])
    dataset_matches = len(source_match_ids)
    split_ready = all(int(split_counts.get(name, 0) or 0) > 0 for name in ("train", "validation", "test"))
    dataset_fresh = bool(manifest) and dataset_matches >= completed_matches and sample_count > 0
    dataset_sha256 = (manifest or {}).get("samples_sha256")

    training_runs = store.list_training_runs(limit=1000)
    latest_training = training_runs[0] if training_runs else None
    latest_training_uses_dataset = bool(
        latest_training and dataset_sha256
        and latest_training.get("dataset_id") == dataset_sha256
    )

    candidate_path = Path(candidate_path)
    candidate_exists = candidate_path.is_file()
    candidate_sha256 = file_sha256(candidate_path) if candidate_exists else None
    selected_model_path = Path(selected_model_path)
    selected = _read_json(selected_model_path) if selected_model_path.is_file() else None
    selected_sha256 = (selected or {}).get("sha256")
    candidate_matches_selected = bool(
        candidate_sha256 and selected_sha256
        and str(candidate_sha256).lower() == str(selected_sha256).lower()
    )

    gate = _read_json(candidate_gate_path) if candidate_gate_path else None
    promotion_gate_passed = bool(gate and gate.get("passed") is True)
    trust_models = store.list_trust_models(limit=1000)
    trust_model_ready = (
        len(applied) >= minimum_applied
        and positive >= minimum_per_class
        and negative >= minimum_per_class
    )
    candidate_training_ready = bool(
        dataset_fresh and split_ready and sample_count > 0
    )
    formal_benchmark_ready = bool(candidate_training_ready and candidate_exists)
    promotion_ready = bool(formal_benchmark_ready and promotion_gate_passed)

    blockers = []
    if len(applied) < minimum_applied:
        blockers.append({
            "gate": "trust_samples",
            "observed": len(applied),
            "required": minimum_applied,
        })
    if positive < minimum_per_class:
        blockers.append({
            "gate": "positive_feedback",
            "observed": positive,
            "required": minimum_per_class,
        })
    if negative < minimum_per_class:
        blockers.append({
            "gate": "negative_feedback",
            "observed": negative,
            "required": minimum_per_class,
        })
    if not manifest:
        blockers.append({"gate": "dataset", "reason": "dataset manifest missing"})
    elif dataset_matches < completed_matches:
        blockers.append({
            "gate": "dataset_freshness",
            "observed_source_matches": dataset_matches,
            "completed_matches": completed_matches,
        })
    if manifest and not split_ready:
        blockers.append({
            "gate": "dataset_splits",
            "observed": split_counts,
            "required_nonzero": ["train", "validation", "test"],
        })
    if not candidate_exists:
        blockers.append({"gate": "candidate_model", "reason": "candidate file missing"})
    if candidate_exists and not promotion_gate_passed:
        blockers.append({"gate": "promotion_report", "reason": "candidate gate has not passed"})

    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "database": {
            "counts": counts,
            "completed_matches": completed_matches,
            "retrieval_outcomes": len(outcomes),
            "applied_outcomes": len(applied),
            "positive_feedback": positive,
            "negative_feedback": negative,
        },
        "feedback_gate": {
            "minimum_applied": int(minimum_applied),
            "minimum_per_class": int(minimum_per_class),
            "passed": trust_model_ready,
        },
        "dataset": {
            "directory": str(dataset_dir.resolve()),
            "manifest_present": manifest is not None,
            "sample_count": sample_count,
            "split_counts": split_counts,
            "source_matches": dataset_matches,
            "completed_matches": completed_matches,
            "fresh": dataset_fresh,
            "splits_ready": split_ready,
            "samples_sha256": dataset_sha256,
        },
        "candidate": {
            "path": str(candidate_path.resolve()),
            "exists": candidate_exists,
            "sha256": candidate_sha256,
            "latest_training_dataset": (
                latest_training.get("dataset_id") if latest_training else None
            ),
            "latest_training_uses_current_dataset": latest_training_uses_dataset,
            "matches_selected_model": candidate_matches_selected,
        },
        "trust_models": {
            "count": len(trust_models),
            "configured_path": str(Path(trust_model_path).resolve()) if trust_model_path else None,
            "configured_file_exists": bool(trust_model_path and Path(trust_model_path).is_file()),
        },
        "promotion": {
            "gate_report_present": gate is not None,
            "gate_passed": promotion_gate_passed,
        },
        "ready_for_trust_training": trust_model_ready,
        "ready_for_candidate_training": candidate_training_ready,
        "ready_for_formal_benchmark": formal_benchmark_ready,
        "ready_for_promotion": promotion_ready,
        "blockers": blockers,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(memory_db_path()))
    parser.add_argument(
        "--dataset-dir",
        default=str(WORKSPACE_ROOT / "evaluation" / "memory" / "dataset_bootstrap"),
    )
    parser.add_argument(
        "--candidate",
        default=str(WORKSPACE_ROOT / "artifacts" / "memory" / "candidates" / "hex_model_v5_memory_candidate.pt"),
    )
    parser.add_argument(
        "--selected-model",
        default=str(WORKSPACE_ROOT / "models" / "selected_model.json"),
    )
    parser.add_argument(
        "--candidate-gate",
        default=str(WORKSPACE_ROOT / "evaluation" / "memory" / "candidate_gate_smoke.json"),
    )
    parser.add_argument("--trust-model")
    parser.add_argument("--minimum-applied", type=int, default=20)
    parser.add_argument("--minimum-per-class", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    with ExperienceStore(args.db) as store:
        report = build_readiness_report(
            store,
            dataset_dir=args.dataset_dir,
            candidate_path=args.candidate,
            selected_model_path=args.selected_model,
            candidate_gate_path=args.candidate_gate,
            trust_model_path=args.trust_model,
            minimum_applied=args.minimum_applied,
            minimum_per_class=args.minimum_per_class,
        )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
