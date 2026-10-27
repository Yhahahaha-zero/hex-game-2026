# -*- coding: utf-8 -*-
"""Run fixed-seed AB/BA evaluation for V1 memory-assisted AI."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

TOOLS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_ROOT.parent
for path in (str(PROJECT_ROOT), str(TOOLS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ai.memory.schema import memory_db_path
from benchmark_ai import DEFAULT_OUTPUT_DIR, run_benchmark


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ai", default="hybrid")
    parser.add_argument("--mode", choices=("shadow", "assist"), default="shadow")
    parser.add_argument("--db", default=str(memory_db_path()))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--minimum-similarity", type=float, default=0.72)
    parser.add_argument("--minimum-support", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument(
        "--retrieval-types",
        default="exact,canonical,similar",
        help="comma-separated exact,canonical,similar",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    os.environ["HEX_MEMORY_RETRIEVAL"] = args.mode
    os.environ["HEX_MEMORY_DB"] = args.db
    os.environ["HEX_MEMORY_TOP_K"] = str(args.top_k)
    os.environ["HEX_MEMORY_MIN_SIMILARITY"] = str(args.minimum_similarity)
    os.environ["HEX_MEMORY_MIN_SUPPORT"] = str(args.minimum_support)
    os.environ["HEX_MEMORY_ALPHA"] = str(args.alpha)
    os.environ["HEX_MEMORY_RETRIEVAL_TYPES"] = args.retrieval_types
    memory_spec = f"memory-{args.mode}:{args.base_ai}"
    document, paths = run_benchmark(
        memory_spec,
        args.base_ai,
        games=args.games,
        repeats=args.repeats,
        seed=args.seed,
        output_dir=args.output_dir,
        name=args.name or f"memory_{args.mode}_{args.base_ai}",
        quiet=args.quiet,
    )
    if args.quiet:
        print(document["summary"])
    print(paths["json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())