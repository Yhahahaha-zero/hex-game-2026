# -*- coding: utf-8 -*-
"""Run the explicit V4 learning-orchestrator cycle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.memory.orchestrator import LearningOrchestrator, OrchestratorConfig
from ai.memory.schema import memory_db_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(memory_db_path()))
    parser.add_argument("--dataset-output", required=True)
    parser.add_argument("--candidate-output", required=True)
    parser.add_argument("--parent-checkpoint", default="none")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minimum-matches", type=int, default=20)
    parser.add_argument("--minimum-samples", type=int, default=1000)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    orchestrator = LearningOrchestrator(
        OrchestratorConfig(
            minimum_new_matches=args.minimum_matches,
            minimum_training_samples=args.minimum_samples,
            auto_train=bool(args.train),
            auto_promote=False,
        ),
        db_path=args.db,
    )
    result = orchestrator.run(
        dataset_dir=args.dataset_output,
        candidate_output=args.candidate_output,
        parent_checkpoint=args.parent_checkpoint,
        epochs=args.epochs,
        train=args.train,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result.get("status") != "not_ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())