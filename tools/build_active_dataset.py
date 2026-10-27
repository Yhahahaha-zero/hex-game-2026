# -*- coding: utf-8 -*-
"""Build a V4 active-learning candidate queue from retrieval outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.memory.active_learning import persist_candidates, score_candidate, select_candidates
from ai.memory.schema import memory_db_path
from ai.memory.store import ExperienceStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(memory_db_path()))
    parser.add_argument("--quota", type=int, default=200)
    parser.add_argument("--dataset-id")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    with ExperienceStore(args.db) as store:
        events = {
            row["retrieval_event_id"]: row
            for row in store.list_retrieval_events(limit=1000000)
        }
        scores = []
        for outcome in store.list_retrieval_outcomes(limit=1000000):
            event = events.get(outcome["retrieval_event_id"])
            if event is None or not event.get("selected_memory_decision_id"):
                continue
            evidence = json.loads(event.get("evidence_json") or "{}")
            scores.append(score_candidate(
                position_id=event["query_position_id"],
                decision_id=event["selected_memory_decision_id"],
                trust_probability=float(event.get("selection_probability") or 0.5),
                confidence=float(event.get("confidence") or 0.0),
                support_count=int(evidence.get("support_count", 0) or 0),
                conflict=bool(evidence.get("conflict", False)),
                retrieval_type=(evidence.get("hits") or [{}])[0].get(
                    "similarity_type", "similar"
                ),
                ply=int(event.get("query_ply") or 1),
            ))
        selected = select_candidates(scores, quota=args.quota)
        persisted = persist_candidates(store, selected, dataset_id=args.dataset_id)
    document = {
        "selected": [score.__dict__ for score in selected],
        "persisted": persisted,
    }
    text = json.dumps(document, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())