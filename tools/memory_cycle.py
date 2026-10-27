# -*- coding: utf-8 -*-
"""Run an explicit V3 validation/features/dataset/training cycle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.memory.dataset import build_memory_dataset
from ai.memory.dataset_registry import register_dataset_version
from ai.memory.pipeline import MemoryLearningPipeline
from ai.memory.schema import memory_db_path
from ai.memory.store import ExperienceStore
from ai.train_memory import train_memory_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(memory_db_path()))
    parser.add_argument("--dataset-output")
    parser.add_argument("--candidate-output")
    parser.add_argument("--parent-checkpoint", default="none")
    parser.add_argument("--training-summary")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--skip-finalize", action="store_true")
    parser.add_argument("--export-dataset", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args(argv)
    result = {"database": str(Path(args.db).resolve())}
    if not args.skip_finalize:
        pipeline = MemoryLearningPipeline(
            lambda: ExperienceStore(args.db).open(), mode="full"
        )
        result["finalized"] = pipeline.process_pending(limit=1000)
    dataset_version = None
    if args.export_dataset:
        if not args.dataset_output:
            raise ValueError("--dataset-output is required with --export-dataset")
        with ExperienceStore(args.db) as store:
            export = build_memory_dataset(
                store, args.dataset_output, seed=args.seed
            )
            dataset_version = register_dataset_version(store, args.dataset_output)
        result["dataset"] = {
            "directory": str(export.output_dir),
            "samples": export.sample_count,
            "sha256": export.sha256,
            "dataset_id": dataset_version["dataset_id"],
        }
    if args.train:
        if not args.dataset_output:
            raise ValueError("--dataset-output is required with --train")
        train_result = train_memory_model(
            dataset_dir=args.dataset_output,
            output_path=args.candidate_output or (
                PROJECT_ROOT.parent / "models" / "hex_model_v5_memory_candidate.pt"
            ),
            parent_checkpoint=args.parent_checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            lineage_db=args.db,
            summary_path=args.training_summary or (
                PROJECT_ROOT.parent / "evaluation" / "training"
                / "memory_training_summary.json"
            ),
        )
        result["training"] = train_result
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())