# -*- coding: utf-8 -*-
"""Shadow-mode match collector for Hex experience memory.

The collector never influences move selection. It buffers a complete two-game
match in memory and submits one immutable snapshot after an explicit accept
boundary. SQLite writes run on a dedicated writer thread.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
from pathlib import Path
import queue
import threading
import time

from ai.memory.canonical import canonicalize_position
from ai.memory.ids import (
    build_decision_id,
    build_game_id,
    build_memory_position_id,
    new_match_id,
)
from ai.memory.schema import (
    COLLECTOR_VERSION,
    FEATURE_VERSION,
    RULES_VERSION,
    memory_db_path,
)
from ai.memory.store import (
    DecisionRecord,
    ExperienceStore,
    GameExperience,
    GameRecord,
    MatchExperience,
    MatchRecord,
    PositionRecord,
)
from ai.memory.trace import (
    ActorMetadata,
    board_state_json,
    json_text,
    snapshot_board,
)
from hex_board import BLUE, RED


LOGGER = logging.getLogger(__name__)
CAPTURE_OFF = "off"
CAPTURE_SHADOW = "shadow"
CAPTURE_MODES = (CAPTURE_OFF, CAPTURE_SHADOW)


class MemoryCollectorError(Exception):
    """Raised for invalid collector state or event sequencing."""


@dataclass(frozen=True)
class MemoryCaptureConfig:
    mode: str = CAPTURE_OFF
    db_path: str | None = None
    queue_size: int = 32
    flush_timeout_seconds: float = 3.0
    fail_open: bool = True


@dataclass(frozen=True)
class MemoryCollectorStats:
    buffered_moves: int = 0
    queued_matches: int = 0
    active_writes: int = 0
    accepted: int = 0
    written: int = 0
    failed: int = 0
    dropped: int = 0


@dataclass
class _CollectedMove:
    position: PositionRecord
    trace: object
    decision_id: str
    decision_created_at: str


@dataclass
class _GameBuffer:
    game_no: int
    sides: dict
    actor_types: dict
    actors: dict
    started_at: str
    moves: list = field(default_factory=list)
    winner_color: int | None = None
    termination_reason: str | None = None
    ended_at: str | None = None


@dataclass
class _MatchBuffer:
    match_id: str
    mode_key: int | None
    mode_name: str
    red_team: str
    blue_team: str
    source_type: str
    started_at: str
    games: list = field(default_factory=list)


def _now():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _parse_non_negative_int(value, default):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _parse_positive_float(value, default):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _parse_bool(value, default=True):
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() not in ("0", "false", "no", "off")


def _read_env_file(path):
    values = {}
    if not Path(path).is_file():
        return values
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        LOGGER.warning("Cannot read memory env file %s: %s", path, exc)
        return values
    for line_no, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        name = name.strip()
        if not separator or not name or not name.replace("_", "").isalnum():
            LOGGER.warning("Ignoring malformed memory env setting at line %s", line_no)
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[name] = value
    return values


def _env_setting(file_values, environ, name, default=""):
    if name in environ:
        return str(environ[name]).strip()
    return str(file_values.get(name, default)).strip()


def create_memory_collector_from_env(*, store_factory=None, env_file=None,
                                     environ=None):
    default_env = Path(__file__).resolve().parents[2] / ".env.local"
    file_values = _read_env_file(Path(env_file) if env_file is not None else default_env)
    environment = os.environ if environ is None else environ
    mode = _env_setting(
        file_values, environment, "HEX_MEMORY_CAPTURE", CAPTURE_OFF
    ).lower()
    if mode not in CAPTURE_MODES:
        LOGGER.warning("Unknown HEX_MEMORY_CAPTURE=%r; capture disabled", mode)
        mode = CAPTURE_OFF
    if mode == CAPTURE_OFF:
        return NullMemoryCollector()
    config = MemoryCaptureConfig(
        mode=mode,
        db_path=(
            _env_setting(file_values, environment, "HEX_MEMORY_DB", "")
            or str(memory_db_path(environment))
        ),
        queue_size=_parse_non_negative_int(
            _env_setting(file_values, environment, "HEX_MEMORY_QUEUE_SIZE", "32"), 32
        ),
        flush_timeout_seconds=_parse_positive_float(
            _env_setting(
                file_values,
                environment,
                "HEX_MEMORY_FLUSH_TIMEOUT_SECONDS",
                "3",
            ),
            3.0,
        ),
        fail_open=_parse_bool(
            _env_setting(file_values, environment, "HEX_MEMORY_FAIL_OPEN", "1"),
            True,
        ),
    )
    return MemoryCollector(config, store_factory=store_factory)


class NullMemoryCollector:
    """No-op collector used for the default off mode."""

    enabled = False

    def start_match(self, **kwargs):
        return None

    def begin_game(self, *args, **kwargs):
        return None

    def record_move(self, *args, **kwargs):
        return None

    def mark_game_over(self, *args, **kwargs):
        return None

    def undo_moves(self, count):
        return 0

    def accept_match(self, **kwargs):
        return False

    def abort_match(self, reason):
        return None

    def flush(self, timeout=None):
        return True

    def close(self):
        return None

    def stats(self):
        return MemoryCollectorStats()

    @property
    def current_context(self):
        return {}

class MemoryCollector:
    """Buffer complete matches and persist them without affecting gameplay."""

    enabled = True

    def __init__(self, config, *, store_factory=None, clock=None, synchronous=False):
        if not isinstance(config, MemoryCaptureConfig):
            raise TypeError("config must be MemoryCaptureConfig")
        if config.mode != CAPTURE_SHADOW:
            raise ValueError("MemoryCollector requires shadow capture mode")
        self.config = config
        self._store_factory = store_factory or self._default_store_factory
        self._clock = clock or _now
        self._synchronous = bool(synchronous)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._queue = queue.Queue(maxsize=config.queue_size)
        self._writer_thread = None
        self._writer_stop = object()
        self._active_writes = 0
        self._stats = {
            "accepted": 0,
            "written": 0,
            "failed": 0,
            "dropped": 0,
        }
        self._match = None
        self._current_game = None
        self.last_error = None

    def _default_store_factory(self):
        return ExperienceStore(self.config.db_path).open()

    def _require_match(self):
        if self._match is None:
            raise MemoryCollectorError("No active match")
        return self._match

    def _require_game(self):
        if self._current_game is None:
            raise MemoryCollectorError("No active game")
        return self._current_game

    def start_match(self, *, mode_key, mode_name, red_team, blue_team,
                    source_type, actor_types, actor_metadata=None,
                    started_at=None):
        with self._lock:
            if self._match is not None:
                self._clear_match()
            timestamp = started_at or self._clock()
            self._match = _MatchBuffer(
                match_id=new_match_id(),
                mode_key=mode_key,
                mode_name=str(mode_name),
                red_team=str(red_team),
                blue_team=str(blue_team),
                source_type=str(source_type),
                started_at=timestamp,
            )
            self.begin_game(
                1,
                sides={RED: red_team, BLUE: blue_team},
                actor_types=actor_types,
                actor_metadata=actor_metadata,
                started_at=timestamp,
            )
            return self._match.match_id

    def begin_game(self, game_no, *, sides, actor_types, actor_metadata=None,
                   started_at=None):
        with self._lock:
            match = self._require_match()
            if game_no not in (1, 2):
                raise MemoryCollectorError("game_no must be 1 or 2")
            if self._current_game is not None and self._current_game.winner_color is None:
                raise MemoryCollectorError("Cannot begin a new game before game over")
            if match.games and game_no != match.games[-1].game_no + 1:
                raise MemoryCollectorError("Games must be started in order")
            if game_no == 1 and match.games:
                raise MemoryCollectorError("Game 1 has already been started")
            actors = {RED: None, BLUE: None}
            if actor_metadata:
                actors.update(actor_metadata)
            game = _GameBuffer(
                game_no=game_no,
                sides={RED: sides[RED], BLUE: sides[BLUE]},
                actor_types={RED: actor_types[RED], BLUE: actor_types[BLUE]},
                actors=actors,
                started_at=started_at or self._clock(),
            )
            match.games.append(game)
            self._current_game = game
            return game

    def record_move(self, board_before, trace, *, occurred_at=None):
        with self._lock:
            game = self._require_game()
            if game.winner_color is not None:
                raise MemoryCollectorError("Cannot record a move after game over")
            if trace.color not in (RED, BLUE):
                raise MemoryCollectorError("Invalid move color")
            if trace.actor.actor_type != game.actor_types[trace.color]:
                raise MemoryCollectorError("Actor type does not match the game side")
            if not isinstance(trace.move, tuple) or len(trace.move) != 2:
                raise MemoryCollectorError("Invalid trace move")
            q, r = trace.move
            if not board_before.in_range(q, r) or not board_before.is_empty(q, r):
                raise MemoryCollectorError("Trace move is not legal on the snapshot")
            snapshot = snapshot_board(board_before)
            ply = len(game.moves) + 1
            position_id = build_memory_position_id(snapshot, trace.color)
            canonical = canonicalize_position(snapshot, trace.color)
            created_at = occurred_at or self._clock()
            position = PositionRecord(
                position_id=position_id,
                experience_key=position_id,
                canonical_key=canonical.key,
                board_json=board_state_json(snapshot),
                current_color=trace.color,
                ply=ply,
                created_at=created_at,
                rules_version=RULES_VERSION,
                feature_version=FEATURE_VERSION,
            )
            game_id = build_game_id(self._match.match_id, game.game_no)
            decision_id = build_decision_id(
                game_id, ply, trace.actor.actor_type, trace.color
            )
            game.actors[trace.color] = trace.actor
            game.moves.append(_CollectedMove(
                position=position,
                trace=trace,
                decision_id=decision_id,
                decision_created_at=created_at,
            ))
            return decision_id

    def mark_game_over(self, winner_color, reason, *, ended_at=None):
        with self._lock:
            game = self._require_game()
            if winner_color not in (RED, BLUE):
                raise MemoryCollectorError("Invalid winner color")
            if not game.moves:
                raise MemoryCollectorError("A completed game must contain moves")
            if game.winner_color is not None:
                if game.winner_color != winner_color:
                    raise MemoryCollectorError("Conflicting winner color")
                return
            game.winner_color = winner_color
            game.termination_reason = str(reason or "connect")
            game.ended_at = ended_at or self._clock()

    def undo_moves(self, count):
        with self._lock:
            game = self._require_game()
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise MemoryCollectorError("count must be a positive integer")
            actual = min(count, len(game.moves))
            if actual:
                del game.moves[-actual:]
            game.winner_color = None
            game.termination_reason = None
            game.ended_at = None
            return actual

    def _actor_for(self, game, color):
        actor = game.actors.get(color)
        if actor is not None:
            return actor
        return ActorMetadata(actor_type=game.actor_types[color])

    def _build_experience(self):
        match = self._require_match()
        if len(match.games) != 2:
            raise MemoryCollectorError("A complete match requires exactly two games")
        game_items = []
        ended_times = []
        for game in match.games:
            if game.winner_color not in (RED, BLUE) or game.ended_at is None:
                raise MemoryCollectorError("Every game must be completed before acceptance")
            if not game.moves:
                raise MemoryCollectorError("Every game must contain at least one move")
            ended_times.append(game.ended_at)
            red = self._actor_for(game, RED)
            blue = self._actor_for(game, BLUE)
            game_id = build_game_id(match.match_id, game.game_no)
            game_record = GameRecord(
                game_id=game_id,
                match_id=match.match_id,
                game_no=game.game_no,
                red_actor_type=game.actor_types[RED],
                blue_actor_type=game.actor_types[BLUE],
                red_actor_id=red.actor_id,
                blue_actor_id=blue.actor_id,
                red_ai_name=red.ai_name,
                blue_ai_name=blue.ai_name,
                red_ai_version=red.ai_version,
                blue_ai_version=blue.ai_version,
                winner_color=game.winner_color,
                termination_reason=game.termination_reason,
                status="completed",
                started_at=game.started_at,
                ended_at=game.ended_at,
            )
            decisions = []
            for move in game.moves:
                trace = move.trace
                actor = trace.actor
                result = 1.0 if trace.color == game.winner_color else 0.0
                decision = DecisionRecord(
                    decision_id=move.decision_id,
                    game_id=game_id,
                    position_id=move.position.position_id,
                    ply=move.position.ply,
                    actor_type=actor.actor_type,
                    actor_color=trace.color,
                    created_at=move.decision_created_at,
                    actor_id=actor.actor_id or (
                        f"player{1 if trace.color == RED else 2}"
                        if actor.actor_type == "human" else None
                    ),
                    ai_name=actor.ai_name,
                    ai_version=actor.ai_version,
                    candidate_json=json_text(trace.candidate_payload),
                    search_config_json=json_text(trace.search_config),
                    chosen_q=trace.move[0],
                    chosen_r=trace.move[1],
                    result=result,
                    decision_quality="unverified",
                    training_eligible=0,
                    integrity_status="valid",
                    rule_status="valid",
                    result_status="known",
                    validation_reason="complete_match_shadow_capture",
                    validator_version=COLLECTOR_VERSION,
                    model_sha256=actor.model_sha256,
                )
                decisions.append((move.position, decision))
            game_items.append(GameExperience(game=game_record, decisions=tuple(decisions)))
        match_record = MatchRecord(
            match_id=match.match_id,
            mode_name=match.mode_name,
            red_team=match.red_team,
            blue_team=match.blue_team,
            source_type=match.source_type,
            started_at=match.started_at,
            mode_key=match.mode_key,
            ended_at=max(ended_times),
            status="completed",
            capture_mode="shadow",
            collector_version=COLLECTOR_VERSION,
        )
        return MatchExperience(match=match_record, games=tuple(game_items))

    def accept_match(self, *, record_id=None):
        with self._lock:
            if self._match is None:
                return False
            experience = self._build_experience()
            if record_id:
                experience = MatchExperience(
                    match=MatchRecord(
                        match_id=experience.match.match_id,
                        mode_name=experience.match.mode_name,
                        red_team=experience.match.red_team,
                        blue_team=experience.match.blue_team,
                        source_type=experience.match.source_type,
                        started_at=experience.match.started_at,
                        mode_key=experience.match.mode_key,
                        ended_at=experience.match.ended_at,
                        status=experience.match.status,
                        record_id=str(record_id),
                        capture_mode=experience.match.capture_mode,
                        collector_version=experience.match.collector_version,
                    ),
                    games=experience.games,
                )
            self._stats["accepted"] += 1
            if self._synchronous:
                succeeded = self._persist_experience(experience)
            else:
                succeeded = self._enqueue_experience(experience)
            self._clear_match()
            return succeeded

    def _enqueue_experience(self, experience):
        self._ensure_writer()
        try:
            self._queue.put_nowait(experience)
        except queue.Full:
            self._stats["dropped"] += 1
            self.last_error = "memory queue is full"
            LOGGER.warning("Memory queue is full; dropping completed match")
            self._condition.notify_all()
            return False
        return True

    def _persist_experience(self, experience, store=None):
        owns_store = store is None
        try:
            if store is None:
                store = self._store_factory()
            store.save_complete_match(experience)
            self._stats["written"] += 1
            return True
        except Exception as exc:
            self._stats["failed"] += 1
            self.last_error = str(exc)
            LOGGER.warning("Failed to persist complete memory match: %s", exc)
            if not self.config.fail_open and self._synchronous:
                raise
            return False
        finally:
            if owns_store and store is not None:
                try:
                    store.close()
                except Exception:
                    pass

    def _ensure_writer(self):
        if self._synchronous:
            return
        if self._writer_thread is not None and self._writer_thread.is_alive():
            return
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="hex-memory-writer",
            daemon=True,
        )
        self._writer_thread.start()

    def _writer_loop(self):
        store = None
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is self._writer_stop:
                        return
                    with self._condition:
                        self._active_writes += 1
                    self._persist_experience(item, store=store)
                finally:
                    with self._condition:
                        self._active_writes = max(0, self._active_writes - 1)
                        self._condition.notify_all()
                    self._queue.task_done()
        finally:
            if store is not None:
                try:
                    store.close()
                except Exception:
                    pass

    def flush(self, timeout=None):
        if self._synchronous:
            return True
        timeout = self.config.flush_timeout_seconds if timeout is None else timeout
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while self._queue.unfinished_tasks or self._active_writes:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
        return True

    def abort_match(self, reason):
        with self._lock:
            self._clear_match()
            return str(reason)

    @property
    def current_context(self):
        with self._lock:
            if self._match is None or self._current_game is None:
                return {}
            return {
                "match_id": self._match.match_id,
                "game_id": build_game_id(
                    self._match.match_id, self._current_game.game_no
                ),
                "game_no": self._current_game.game_no,
                "ply": len(self._current_game.moves) + 1,
            }

    def _clear_match(self):
        self._match = None
        self._current_game = None

    def close(self):
        if self._synchronous:
            return
        writer = self._writer_thread
        if writer is None:
            return
        self.flush()
        if writer.is_alive():
            self._queue.put(self._writer_stop)
            writer.join(self.config.flush_timeout_seconds)
        self._writer_thread = None

    def stats(self):
        with self._lock:
            game = self._current_game
            return MemoryCollectorStats(
                buffered_moves=len(game.moves) if game is not None else 0,
                queued_matches=self._queue.qsize(),
                active_writes=self._active_writes,
                accepted=self._stats["accepted"],
                written=self._stats["written"],
                failed=self._stats["failed"],
                dropped=self._stats["dropped"],
            )