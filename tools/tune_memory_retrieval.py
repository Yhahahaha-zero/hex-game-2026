# -*- coding: utf-8 -*-
"""Tune V3 retrieval thresholds from applied Shadow/Assist outcomes."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.memory.retrieval_policy import RetrievalPolicy, save_retrieval_policy
from ai.memory.schema import memory_db_path
from ai.memory.store import ExperienceStore


def _event_map(store):
    return {
        event["retrieval_event_id"]: event
        for event in store.list_retrieval_events(limit=1000000)
    }


def _support(event):
    try:
        evidence = json.loads(event.get("evidence_json") or "{}")
    except (TypeError, ValueError):
        return 0
    return int(evidence.get("support_count", 0) or 0)


def _score(rows, events, *, confidence, similarity, support):
    selected = []
    for outcome in rows:
        if not int(outcome["applied"]):
            continue
        event = events.get(outcome["retrieval_event_id"])
        if event is None:
            continue
        if float(event.get("confidence") or 0.0) < confidence:
            continue
        if float(event.get("similarity") or 0.0) < similarity:
            continue
        if _support(event) < support:
            continue
        selected.append(float(outcome["retrieval_reward"]))
    coverage = len(selected) / len(rows) if rows else 0.0
    mean_reward = sum(selected) / len(selected) if selected else 0.0
    return {
        "selected": len(selected),
        "coverage": round(coverage, 6),
        "mean_reward": round(mean_reward, 6),
        "objective": round(0.7 * mean_reward + 0.3 * coverage, 6),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(memory_db_path()))
    parser.add_argument("--policy-file", default=str(
        PROJECT_ROOT.parent / "models" / "memory_policy.json"
    ))
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--retrieval-types", default="exact,canonical,similar")
    parser.add_argument("--minimum-coverage", type=float, default=0.0)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    with ExperienceStore(args.db) as store:
        rows = store.list_retrieval_outcomes(limit=1000000)
        events = _event_map(store)
    if not rows:
        raise SystemExit("No retrieval outcomes are available; run shadow/assist games first")
    best = None
    for confidence in (0.45, 0.55, 0.65, 0.75):
        for similarity in (0.60, 0.72, 0.82):
            for support in (2, 3, 5):
                metrics = _score(
                    rows,
                    events,
                    confidence=confidence,
                    similarity=similarity,
                    support=support,
                )
                if metrics["coverage"] < args.minimum_coverage:
                    continue
                candidate = {
                    "minimum_confidence": confidence,
                    "minimum_similarity": similarity,
                    "minimum_support": support,
                    **metrics,
                }
                key = (
                    candidate["objective"],
                    candidate["mean_reward"],
                    candidate["coverage"],
                    -support,
                )
                if best is None or key > best[0]:
                    best = (key, candidate)
    if best is None:
        raise SystemExit("No retrieval policy met the requested coverage gate")
    selected = best[1]
    types = tuple(
        item.strip().lower()
        for item in args.retrieval_types.split(",")
        if item.strip() in ("exact", "canonical", "similar")
    ) or ("exact", "canonical", "similar")
    policy = RetrievalPolicy(
        top_k=5,
        minimum_similarity=selected["minimum_similarity"],
        minimum_support=selected["minimum_support"],
        minimum_confidence=selected["minimum_confidence"],
        alpha=args.alpha,
        retrieval_types=types,
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        benchmark_file=None,
    )
    result = {"selected": selected, "policy": policy.as_dict(), "applied": args.apply}
    if args.apply:
        result["policy_file"] = str(save_retrieval_policy(args.policy_file, policy))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())