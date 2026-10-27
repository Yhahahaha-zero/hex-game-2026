# -*- coding: utf-8 -*-
"""Aggregate V3 memory, dataset, job and model status."""
from __future__ import annotations

import json
from pathlib import Path

from ai.memory.outcomes import summarize_outcomes
from ai.memory.retrieval_policy import load_retrieval_policy


def build_memory_status(store, *, models_dir, policy_file=None):
    counts = store.counts()
    outcomes = store.list_retrieval_outcomes(limit=100000)
    datasets = store.list_dataset_versions(limit=20)
    trainings = store.list_training_runs(limit=20)
    jobs = store.list_memory_jobs(limit=10000)
    models_dir = Path(models_dir)
    selected_path = models_dir / "selected_model.json"
    selected_model = None
    if selected_path.is_file():
        selected_model = json.loads(selected_path.read_text(encoding="utf-8-sig"))
    policy = None
    policy_path = Path(policy_file) if policy_file else models_dir / "memory_policy.json"
    if policy_path.is_file():
        policy = load_retrieval_policy(policy_path).as_dict()
    history_path = models_dir / "selected_model_history.jsonl"
    history_count = 0
    if history_path.is_file():
        history_count = sum(
            1 for line in history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    job_counts = {}
    for job in jobs:
        job_counts[job["status"]] = job_counts.get(job["status"], 0) + 1
    return {
        "schema_version": 1,
        "counts": counts,
        "outcomes": summarize_outcomes(outcomes),
        "jobs": {
            "total": len(jobs),
            "by_status": job_counts,
            "latest": jobs[:10],
        },
        "datasets": datasets,
        "training_runs": trainings,
        "selected_model": selected_model,
        "rollback_available": history_count > 0,
        "promotion_history_count": history_count,
        "retrieval_policy": policy,
        "retrieval_policy_file": str(policy_path) if policy else None,
    }