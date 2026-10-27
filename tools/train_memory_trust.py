# -*- coding: utf-8 -*-
"""Train a calibrated V4 memory trust model from retrieval outcomes."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.memory.schema import memory_db_path
from ai.memory.store import ExperienceStore, TrustModelRecord
from ai.memory.trust import MemoryTrustFeatures, MemoryTrustModel, file_sha256, save_trust_model


def _features(event, outcome):
    evidence = json.loads(event.get("evidence_json") or "{}")
    hits = evidence.get("hits", [])
    top = hits[0] if hits else {}
    return MemoryTrustFeatures(
        similarity=float(outcome.get("similarity") or 0.0),
        confidence=float(outcome.get("confidence") or 0.0),
        support_count=int(evidence.get("support_count", 0) or 0),
        vote_share=float(evidence.get("vote_share", 0.0) or 0.0),
        conflict=bool(evidence.get("conflict", False)),
        ply=int(event.get("query_ply") or 1),
        retrieval_type=str(top.get("similarity_type", "similar")),
        model_agrees=bool(event.get("base_q") == event.get("memory_q")
                          and event.get("base_r") == event.get("memory_r")),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(memory_db_path()))
    parser.add_argument("--output", default=str(
        PROJECT_ROOT.parent / "models" / "memory_trust_model.json"
    ))
    parser.add_argument("--epochs", type=int, default=400)
    args = parser.parse_args(argv)
    with ExperienceStore(args.db) as store:
        outcomes = [
            row for row in store.list_retrieval_outcomes(limit=1000000)
            if int(row["applied"])
        ]
        events = {
            row["retrieval_event_id"]: row
            for row in store.list_retrieval_events(limit=1000000)
        }
        features = []
        labels = []
        for outcome in outcomes:
            event = events.get(outcome["retrieval_event_id"])
            if event is None:
                continue
            features.append(_features(event, outcome))
            labels.append(1 if float(outcome["retrieval_reward"]) > 0 else 0)
        if not features:
            raise SystemExit("No applied retrieval outcomes are available")
        model = MemoryTrustModel.fit(features, labels, epochs=args.epochs)
        path = save_trust_model(args.output, model)
        model_id = str(uuid.uuid4())
        store.save_trust_model(TrustModelRecord(
            trust_model_id=model_id,
            model_path=str(Path(path).resolve()),
            sha256=file_sha256(path),
            feature_version=model.feature_version,
            metrics_json=json.dumps(model.metrics, sort_keys=True),
            status="candidate",
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        ))
    print(json.dumps({
        "trust_model_id": model_id,
        "path": str(path),
        "metrics": model.metrics,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())