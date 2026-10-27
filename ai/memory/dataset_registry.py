# -*- coding: utf-8 -*-
"""Dataset and training-run lineage registration for V3."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

from ai.memory.store import DatasetVersionRecord, TrainingRunRecord


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def register_dataset_version(store, dataset_dir):
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_id = str(manifest["samples_sha256"])
    record = DatasetVersionRecord(
        dataset_id=dataset_id,
        dataset_dir=str(dataset_dir.resolve()),
        manifest_sha256=_sha256(manifest_path),
        samples_sha256=str(manifest["samples_sha256"]),
        feature_version=int(manifest.get("feature_version", 1)),
        encoding_version=str(manifest["encoding_version"]),
        sample_count=int(manifest["sample_count"]),
        split_counts_json=json.dumps(
            manifest["split_counts"], sort_keys=True, separators=(",", ":")
        ),
        source_match_count=int(len(manifest["match_ids"])),
        created_at=_now(),
    )
    created = store.save_dataset_version(record)
    return {
        "dataset_id": dataset_id,
        "created": created,
        "dataset_dir": record.dataset_dir,
    }


def register_training_run(store, *, training_run_id, dataset_id,
                          parent_model_sha256, candidate_model_sha256,
                          status, metrics):
    record = TrainingRunRecord(
        training_run_id=str(training_run_id),
        dataset_id=str(dataset_id),
        parent_model_sha256=parent_model_sha256,
        candidate_model_sha256=candidate_model_sha256,
        status=status,
        metrics_json=json.dumps(metrics, ensure_ascii=False, sort_keys=True),
        started_at=_now(),
        finished_at=_now() if status != "running" else None,
    )
    return store.save_training_run(record)