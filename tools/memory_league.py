# -*- coding: utf-8 -*-
"""Run the V4 historical-model league."""
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

from ai.memory.league import OpponentSpec, register_league_run, run_league
from ai.memory.schema import memory_db_path
from ai.memory.store import ExperienceStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--opponent", action="append", required=True,
                        help="id:spec, repeatable")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--db", default=str(memory_db_path()))
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    opponents = []
    for value in args.opponent:
        opponent_id, separator, spec = value.partition(":")
        if not separator or not spec:
            raise ValueError("opponent must use id:spec")
        opponents.append(OpponentSpec(opponent_id, spec, spec))
    document = run_league(
        args.candidate,
        opponents,
        games=args.games,
        repeats=args.repeats,
        seed=args.seed,
    )
    with ExperienceStore(args.db) as store:
        run_id = register_league_run(store, document)
    document["league_run_id"] = run_id
    text = json.dumps(document, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())