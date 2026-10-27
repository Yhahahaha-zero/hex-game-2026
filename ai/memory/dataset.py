# -*- coding: utf-8 -*-
"""Export validated memory into leak-free policy/value training datasets."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from ai.encode import encode_board
from ai.memory.features import FEATURE_VERSION
from ai.model_v5 import ENCODING_VERSION, POLICY_SIZE, get_legal_moves, move_to_index
from hex_board import BLUE, EMPTY, RED, HexBoard


DATASET_SCHEMA_VERSION = 1
DEFAULT_SPLIT_RATIOS = (0.70, 0.15, 0.15)


@dataclass(frozen=True)
class DatasetExport:
    output_dir: Path
    samples_path: Path
    manifest_path: Path
    sample_count: int
    train_count: int
    validation_count: int
    test_count: int
    sha256: str


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _board_from_json(raw_value):
    document = json.loads(raw_value)
    grid = document["grid"]
    if len(grid) != 11 or any(len(row) != 11 for row in grid):
        raise ValueError("invalid board dimensions")
    if any(cell not in (EMPTY, RED, BLUE) for row in grid for cell in row):
        raise ValueError("invalid board value")
    board = HexBoard()
    board.grid = [list(row) for row in grid]
    board.history = []
    return board


def split_match_ids(match_ids, *, seed=20261025, ratios=DEFAULT_SPLIT_RATIOS):
    unique = sorted(set(str(value) for value in match_ids))
    if not unique:
        return {"train": (), "validation": (), "test": ()}
    if len(ratios) != 3 or abs(sum(float(value) for value in ratios) - 1.0) > 1e-6:
        raise ValueError("split ratios must contain three values summing to 1")
    ordered = sorted(
        unique,
        key=lambda value: hashlib.sha256(
            f"{seed}:{value}".encode("utf-8")
        ).hexdigest(),
    )
    count = len(ordered)
    if count == 1:
        sizes = [1, 0, 0]
    elif count == 2:
        sizes = [1, 1, 0]
    else:
        train = max(1, int(count * float(ratios[0])))
        validation = max(1, int(count * float(ratios[1])))
        if train + validation >= count:
            train = max(1, count - 2)
            validation = 1
        test = count - train - validation
        sizes = [train, validation, test]
    train_end = sizes[0]
    validation_end = train_end + sizes[1]
    return {
        "train": tuple(ordered[:train_end]),
        "validation": tuple(ordered[train_end:validation_end]),
        "test": tuple(ordered[validation_end:]),
    }


def _policy_target(board, action_index, smoothing=0.05):
    target = np.zeros(POLICY_SIZE, dtype=np.float32)
    legal = [move_to_index(q, r) for q, r in get_legal_moves(board)]
    if not legal:
        target[action_index] = 1.0
        return target
    for index in legal:
        target[index] = smoothing / len(legal)
    target[action_index] += 1.0 - smoothing
    return target


def _atomic_write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if Path(temp_name).exists():
            Path(temp_name).unlink()


def build_memory_dataset(store, output_dir, *, seed=20261025,
                         split_ratios=DEFAULT_SPLIT_RATIOS,
                         policy_smoothing=0.05, active_only=False,
                         allowed_sources=None):
    rows = store.query_training_samples()
    if isinstance(allowed_sources, str):
        allowed_sources = (allowed_sources,)
    elif allowed_sources is not None:
        allowed_sources = tuple(allowed_sources)
    if allowed_sources:
        allowed = set(allowed_sources)
        rows = [row for row in rows if row.get("source_type") in allowed]
    if active_only:
        active_ids = {
            row["decision_id"] for row in store.list_active_learning_candidates(
                selected=1, limit=1000000
            )
        }
        rows = [row for row in rows if row["decision_id"] in active_ids]
    if not rows:
        raise ValueError("no validated training decisions are available")
    splits = split_match_ids(
        [row["match_id"] for row in rows],
        seed=seed,
        ratios=split_ratios,
    )
    match_ids = tuple(splits["train"] + splits["validation"] + splits["test"])
    match_index = {match_id: index for index, match_id in enumerate(match_ids)}
    split_code = {}
    for code, name in enumerate(("train", "validation", "test")):
        for match_id in splits[name]:
            split_code[match_id] = code
    states = []
    legal_masks = []
    actions = []
    values = []
    policies = []
    match_indices = []
    position_ids = []
    decision_ids = []
    game_ids = []
    skipped = []
    for row in rows:
        try:
            board = _board_from_json(row["board_json"])
            player = int(row["current_color"])
            if player not in (RED, BLUE):
                raise ValueError("invalid player")
            action = move_to_index(int(row["chosen_q"]), int(row["chosen_r"]))
            legal = [move_to_index(q, r) for q, r in get_legal_moves(board)]
            if action not in legal:
                raise ValueError("chosen move is not legal")
            states.append(encode_board(board, player).cpu().numpy().astype(np.float32))
            mask = np.zeros(POLICY_SIZE, dtype=np.bool_)
            mask[legal] = True
            legal_masks.append(mask)
            actions.append(action)
            result = float(row["result"])
            values.append(1.0 if result >= 0.5 else -1.0)
            policies.append(_policy_target(board, action, smoothing=policy_smoothing))
            match_indices.append(match_index[row["match_id"]])
            position_ids.append(str(row["position_id"]))
            decision_ids.append(str(row["decision_id"]))
            game_ids.append(str(row["game_id"]))
        except Exception as exc:
            skipped.append({
                "decision_id": row.get("decision_id"),
                "error": str(exc),
            })
    if not states:
        raise ValueError("all training samples were invalid")
    states_array = np.stack(states)
    masks_array = np.stack(legal_masks)
    actions_array = np.asarray(actions, dtype=np.int64)
    values_array = np.asarray(values, dtype=np.float32)
    policies_array = np.stack(policies)
    matches_array = np.asarray(match_indices, dtype=np.int32)
    split_codes = np.asarray(
        [split_code[match_ids[index]] for index in matches_array], dtype=np.int8
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.npz"
    descriptor, temp_name = tempfile.mkstemp(
        dir=output_dir, prefix=".samples.", suffix=".npz"
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        np.savez_compressed(
            temp_path,
            states=states_array,
            legal_masks=masks_array,
            actions=actions_array,
            value_targets=values_array,
            policy_targets=policies_array,
            match_indices=matches_array,
            split_codes=split_codes,
            position_ids=np.asarray(position_ids),
            decision_ids=np.asarray(decision_ids),
            game_ids=np.asarray(game_ids),
        )
        os.replace(temp_path, samples_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    sha256 = _file_sha256(samples_path)
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "encoding_version": ENCODING_VERSION,
        "feature_version": FEATURE_VERSION,
        "policy_target": "chosen_move_with_legal_smoothing",
        "policy_smoothing": float(policy_smoothing),
        "seed": int(seed),
        "split_ratios": [float(value) for value in split_ratios],
        "match_ids": list(match_ids),
        "splits": {name: list(values) for name, values in splits.items()},
        "sample_count": int(len(actions)),
        "split_counts": {
            "train": int(np.sum(split_codes == 0)),
            "validation": int(np.sum(split_codes == 1)),
            "test": int(np.sum(split_codes == 2)),
        },
        "active_only": bool(active_only),
        "allowed_sources": list(allowed_sources or ()),
        "source_counts": {
            source: sum(1 for row in rows if row.get("source_type") == source)
            for source in sorted({row.get("source_type") for row in rows})
        },
        "skipped_samples": skipped,
        "samples_file": samples_path.name,
        "samples_sha256": sha256,
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_write_json(manifest_path, manifest)
    return DatasetExport(
        output_dir=output_dir,
        samples_path=samples_path,
        manifest_path=manifest_path,
        sample_count=int(len(actions)),
        train_count=int(np.sum(split_codes == 0)),
        validation_count=int(np.sum(split_codes == 1)),
        test_count=int(np.sum(split_codes == 2)),
        sha256=sha256,
    )


def load_memory_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    manifest = json.loads(
        (dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )
    samples_path = dataset_dir / manifest["samples_file"]
    if _file_sha256(samples_path) != manifest["samples_sha256"]:
        raise ValueError("dataset SHA256 mismatch")
    with np.load(samples_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    return arrays, manifest