# -*- coding: utf-8 -*-
"""Historical-model league evaluation and Elo tracking."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import uuid


@dataclass(frozen=True)
class OpponentSpec:
    opponent_id: str
    spec: str
    ai_name: str
    ai_version: str | None = None
    model_sha256: str | None = None
    elo: float = 1500.0


@dataclass(frozen=True)
class LeagueEntry:
    opponent_id: str
    games: int
    wins: int
    losses: int
    win_rate: float
    elo: float


def expected_score(player_elo, opponent_elo):
    return 1.0 / (1.0 + 10.0 ** ((opponent_elo - player_elo) / 400.0))


def update_elo(player_elo, opponent_elo, score, k_factor=24.0):
    return round(
        player_elo + k_factor * (float(score) - expected_score(player_elo, opponent_elo)),
        2,
    )


def build_league_table(results, *, baseline_elo=1500.0):
    rows = []
    elo = baseline_elo
    for result in sorted(results, key=lambda item: str(item["opponent_id"])):
        games = int(result.get("games_completed", 0))
        wins = int(result.get("ai_a_wins", 0))
        losses = int(result.get("ai_b_wins", 0))
        rate = wins / games if games else 0.0
        elo = update_elo(elo, float(result.get("opponent_elo", 1500.0)), rate)
        rows.append(LeagueEntry(
            opponent_id=str(result["opponent_id"]),
            games=games,
            wins=wins,
            losses=losses,
            win_rate=round(rate, 6),
            elo=elo,
        ))
    return tuple(rows)


def run_league(candidate_spec, opponents, *, games=200, repeats=1, seed=20261025,
               output_dir="evaluation/memory/league"):
    from tools import benchmark_ai
    results = []
    for opponent in sorted(opponents, key=lambda item: item.opponent_id):
        document, _paths = benchmark_ai.run_benchmark(
            candidate_spec,
            opponent.spec,
            games=games,
            repeats=repeats,
            seed=seed,
            output_dir=output_dir,
            name=f"league_{opponent.opponent_id}",
            quiet=True,
        )
        row = dict(document["summary"])
        row.update({
            "opponent_id": opponent.opponent_id,
            "opponent_elo": opponent.elo,
            "benchmark": document.get("config", {}),
        })
        results.append(row)
    table = build_league_table(results)
    return {
        "candidate": candidate_spec,
        "results": results,
        "table": [entry.__dict__ for entry in table],
    }


def register_league_run(store, document, *, status="completed"):
    from ai.memory.store import LeagueRunRecord, OpponentProfileRecord
    run_id = str(uuid.uuid4())
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    candidate_ref = str(document.get("candidate", ""))
    candidate_file = candidate_ref
    prefix, separator, suffix = candidate_ref.partition(":")
    is_windows_drive = len(prefix) == 1 and prefix.isalpha()
    if separator and suffix and not is_windows_drive:
        candidate_file = suffix
    candidate_path = Path(candidate_file).expanduser()
    candidate_sha = None
    if candidate_path.is_file():
        candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    store.save_league_run(LeagueRunRecord(
        league_run_id=run_id,
        candidate_sha256=candidate_sha,
        status=status,
        metrics_json=json.dumps(document, ensure_ascii=False, sort_keys=True),
        started_at=now,
        finished_at=now if status != "running" else None,
    ))
    for row in document.get("results", []):
        opponent_id = str(row["opponent_id"])
        opponent = OpponentProfileRecord(
            opponent_id=opponent_id,
            ai_name=str(row.get("ai_b", opponent_id)),
            ai_version=None,
            model_sha256=None,
            elo=float(row.get("opponent_elo", 1500.0)),
            metrics_json=json.dumps(row, ensure_ascii=False, sort_keys=True),
            created_at=now,
        )
        store.save_opponent_profile(opponent)
    return run_id
