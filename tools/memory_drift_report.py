# -*- coding: utf-8 -*-
"""Generate a V4 drift snapshot and report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.memory.drift import build_drift_snapshot
from ai.memory.schema import memory_db_path
from ai.memory.store import ExperienceStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(memory_db_path()))
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    with ExperienceStore(args.db) as store:
        report = build_drift_snapshot(store)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not report["alerts"] else 1


if __name__ == "__main__":
    raise SystemExit(main())