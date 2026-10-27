# -*- coding: utf-8 -*-
"""Export a validated, match-grouped V2 training dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.memory.dataset import build_memory_dataset
from ai.memory.schema import memory_db_path
from ai.memory.store import ExperienceStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(memory_db_path()))
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--policy-smoothing", type=float, default=0.05)
    args = parser.parse_args(argv)
    ratios = (args.train_ratio, args.validation_ratio, args.test_ratio)
    with ExperienceStore(args.db) as store:
        export = build_memory_dataset(
            store,
            args.output,
            seed=args.seed,
            split_ratios=ratios,
            policy_smoothing=args.policy_smoothing,
        )
    print(json.dumps({
        "output_dir": str(export.output_dir),
        "manifest": str(export.manifest_path),
        "samples": export.sample_count,
        "train": export.train_count,
        "validation": export.validation_count,
        "test": export.test_count,
        "sha256": export.sha256,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())