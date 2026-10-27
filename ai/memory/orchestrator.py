# -*- coding: utf-8 -*-
"""Human-controlled V4 learning orchestrator."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import uuid

from ai.memory.dataset import build_memory_dataset
from ai.memory.dataset_registry import register_dataset_version, register_training_run
from ai.memory.schema import memory_db_path
from ai.memory.store import ExperienceStore
from ai.train_memory import train_memory_model


@dataclass(frozen=True)
class OrchestratorConfig:
    minimum_new_matches: int = 20
    minimum_training_samples: int = 1000
    auto_train: bool = False
    auto_promote: bool = False
    dataset_root: str = "artifacts/memory/datasets"


def readiness(store, config):
    matches = store.list_matches(limit=1000000)
    samples = store.query_training_samples(limit=1000000)
    datasets = store.list_dataset_versions(limit=1)
    latest_dataset_count = int(datasets[0]["sample_count"]) if datasets else 0
    new_matches = len(matches)
    return {
        "matches": new_matches,
        "training_samples": len(samples),
        "new_samples_since_dataset": max(0, len(samples) - latest_dataset_count),
        "ready": (
            new_matches >= config.minimum_new_matches
            and len(samples) >= config.minimum_training_samples
        ),
        "auto_train": config.auto_train,
        "auto_promote": False,
    }


class LearningOrchestrator:
    def __init__(self, config=None, *, db_path=None):
        self.config = config or OrchestratorConfig()
        self.db_path = str(db_path or memory_db_path())

    def status(self):
        with ExperienceStore(self.db_path) as store:
            return readiness(store, self.config)

    def create_dataset(self, output_dir):
        with ExperienceStore(self.db_path) as store:
            export = build_memory_dataset(store, output_dir, seed=20261025)
            version = register_dataset_version(store, output_dir)
        return {
            "dataset_id": version["dataset_id"],
            "samples": export.sample_count,
            "sha256": export.sha256,
        }

    def train_candidate(self, *, dataset_dir, candidate_output,
                        parent_checkpoint="none", epochs=5):
        run_id = str(uuid.uuid4())
        summary = train_memory_model(
            dataset_dir=dataset_dir,
            output_path=candidate_output,
            parent_checkpoint=parent_checkpoint,
            epochs=epochs,
            lineage_db=self.db_path,
            training_run_id=run_id,
        )
        return summary

    def run(self, *, dataset_dir, candidate_output, parent_checkpoint="none",
            epochs=5, train=False):
        state = self.status()
        if not state["ready"]:
            return {"status": "not_ready", **state}
        result = {"status": "dataset_ready", **state}
        result["dataset"] = self.create_dataset(dataset_dir)
        if train:
            if not self.config.auto_train:
                raise RuntimeError("auto_train is disabled; pass an explicit training command")
            result["candidate"] = self.train_candidate(
                dataset_dir=dataset_dir,
                candidate_output=candidate_output,
                parent_checkpoint=parent_checkpoint,
                epochs=epochs,
            )
            result["status"] = "awaiting_manual_promotion"
        return result