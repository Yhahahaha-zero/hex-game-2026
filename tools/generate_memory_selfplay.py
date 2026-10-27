# -*- coding: utf-8 -*-
"""Generate explicit two-game self-play matches for V3 experience data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

TOOLS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_ROOT.parent
for path in (str(PROJECT_ROOT), str(TOOLS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ai.memory.pipeline import MemoryLearningPipeline
from ai.memory.schema import memory_db_path
from ai.memory.selfplay import run_selfplay_series
from ai.memory.store import ExperienceStore
from benchmark_ai import load_ai_factory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(memory_db_path()))
    parser.add_argument("--ai-a", default="basic")
    parser.add_argument("--ai-b", default="basic")
    parser.add_argument("--matches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    results = run_selfplay_series(
        load_ai_factory(args.ai_a),
        load_ai_factory(args.ai_b),
        db_path=args.db,
        matches=args.matches,
        seed=args.seed,
    )
    finalized = []
    if args.finalize:
        pipeline = MemoryLearningPipeline(lambda: ExperienceStore(args.db).open(), mode="full")
        finalized = [pipeline.process_match(result.match_id) for result in results]
    document = {
        "database": str(Path(args.db).resolve()),
        "matches": [result.as_dict() for result in results],
        "finalized": finalized,
    }
    text = json.dumps(document, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())