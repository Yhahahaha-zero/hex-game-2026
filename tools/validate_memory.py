# -*- coding: utf-8 -*-
"""Validate and optionally promote recorded Hex experience data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.memory.schema import memory_db_path
from ai.memory.store import ExperienceStore
from ai.memory.validator import validate_all


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(memory_db_path()))
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--allow-invalid", action="store_true")
    args = parser.parse_args(argv)
    with ExperienceStore(args.db) as store:
        reports = validate_all(store, promote=args.promote)
    document = {
        "database": str(Path(args.db).resolve()),
        "promoted": bool(args.promote),
        "matches": len(reports),
        "valid": sum(1 for report in reports if report.valid),
        "invalid": sum(1 for report in reports if not report.valid),
        "eligible_ai_decisions": sum(
            report.ai_decisions_eligible for report in reports
        ),
        "reports": [report.as_dict() for report in reports],
    }
    text = json.dumps(document, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if document["invalid"] == 0 or args.allow_invalid else 1


if __name__ == "__main__":
    raise SystemExit(main())