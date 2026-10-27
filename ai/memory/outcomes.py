# -*- coding: utf-8 -*-
"""Post-game attribution for remembered recommendations."""
from __future__ import annotations

from datetime import datetime
from statistics import mean

from ai.memory.store import RetrievalOutcomeRecord


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def attribute_game_outcomes(store, game_id):
    game = store.get_game(game_id)
    if game is None:
        raise ValueError(f"Unknown game_id: {game_id}")
    events = store.list_retrieval_events(query_game_id=game_id, limit=100000)
    decisions = store.list_decisions(game_id=game_id, limit=100000)
    by_ply = {
        int(row["ply"]): row for row in decisions if row["actor_type"] == "ai"
    }
    created = 0
    for event in events:
        if event.get("query_ply") is None:
            continue
        decision = by_ply.get(int(event["query_ply"]))
        if decision is None or decision.get("result") is None:
            continue
        decision_result = float(decision["result"])
        game_result = 1.0 if decision["actor_color"] == game["winner_color"] else 0.0
        applied = int(event["applied"])
        reward = (1.0 if decision_result >= 0.5 else -1.0) if applied else 0.0
        record = RetrievalOutcomeRecord(
            retrieval_event_id=event["retrieval_event_id"],
            game_id=game_id,
            decision_id=decision["decision_id"],
            applied=applied,
            decision_result=decision_result,
            game_result=game_result,
            retrieval_reward=reward,
            similarity=event.get("similarity"),
            confidence=event.get("confidence"),
            retrieval_mode=event["retrieval_mode"],
            created_at=_now(),
        )
        if store.save_retrieval_outcome(record):
            created += 1
    return created


def attribute_match_outcomes(store, match_id):
    document = store.load_match_for_validation(match_id)
    if document is None:
        raise ValueError(f"Unknown match_id: {match_id}")
    return {
        game["game_id"]: attribute_game_outcomes(store, game["game_id"])
        for game in document["games"]
    }


def summarize_outcomes(rows):
    applied = [row for row in rows if int(row["applied"])]
    rewards = [float(row["retrieval_reward"]) for row in applied]
    coverage = len(applied) / len(rows) if rows else 0.0
    return {
        "outcomes": len(rows),
        "applied": len(applied),
        "coverage": round(coverage, 6),
        "mean_reward": round(mean(rewards), 6) if rewards else 0.0,
        "positive_rate": (
            round(sum(1 for value in rewards if value > 0) / len(rewards), 6)
            if rewards else 0.0
        ),
        "negative_rate": (
            round(sum(1 for value in rewards if value < 0) / len(rewards), 6)
            if rewards else 0.0
        ),
    }