# -*- coding: utf-8 -*-
"""Candidate model promotion gates, atomic selection and rollback."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


class ModelPromotionError(RuntimeError):
    """Raised when a candidate cannot be safely promoted."""


@dataclass(frozen=True)
class PromotionGate:
    minimum_games: int = 1000
    minimum_win_rate: float = 50.0
    minimum_color_win_rate: float = 40.0
    maximum_errors: int = 0
    maximum_move_ms: float = 500.0
    minimum_ci95_low: float | None = None
    maximum_color_gap: float | None = None
    maximum_p95_move_ms: float | None = None


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, document):
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


def validate_candidate_checkpoint(path):
    try:
        import torch
    except ImportError as exc:
        raise ModelPromotionError("PyTorch is required for model promotion") from exc
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ModelPromotionError("Candidate is not a structured checkpoint")
    if checkpoint.get("model_name") not in (None, "", "HexNetV5"):
        raise ModelPromotionError("Candidate model_name is incompatible")
    if checkpoint.get("model_version") not in (None, "", "5.0.0"):
        raise ModelPromotionError("Candidate model_version is incompatible")
    if checkpoint.get("encoding_version") not in (None, "", "3x11x11-v1"):
        raise ModelPromotionError("Candidate encoding_version is incompatible")
    if not checkpoint.get("model_state_dict"):
        raise ModelPromotionError("Candidate state_dict is empty")
    return True


def evaluate_promotion_gate(benchmark_document, gate=None):
    gate = gate or PromotionGate()
    summary = benchmark_document.get("summary", benchmark_document)
    errors = int(summary.get("errors", summary.get("error_rate", 0)) or 0)
    red_rate = float(summary.get("ai_a_red_win_rate", 0.0))
    blue_rate = float(summary.get("ai_a_blue_win_rate", 0.0))
    ci95_low = float(summary.get("ai_a_win_rate_ci95_low", 0.0))
    p95_move_ms = float(summary.get("ai_a_p95_move_ms", float("inf")))
    checks = {
        "games": int(summary.get("games_completed", 0)) >= gate.minimum_games,
        "win_rate": float(summary.get("ai_a_win_rate", 0.0)) >= gate.minimum_win_rate,
        "red_win_rate": red_rate >= gate.minimum_color_win_rate,
        "blue_win_rate": blue_rate >= gate.minimum_color_win_rate,
        "errors": errors <= gate.maximum_errors,
        "latency": (
            float(summary.get("ai_a_max_move_ms", float("inf")))
            <= gate.maximum_move_ms
        ),
    }
    if gate.minimum_ci95_low is not None:
        checks["confidence_lower_bound"] = ci95_low >= gate.minimum_ci95_low
    if gate.maximum_color_gap is not None:
        checks["color_gap"] = abs(red_rate - blue_rate) <= gate.maximum_color_gap
    if gate.maximum_p95_move_ms is not None:
        checks["p95_latency"] = p95_move_ms <= gate.maximum_p95_move_ms
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "gate": {
            "minimum_games": gate.minimum_games,
            "minimum_win_rate": gate.minimum_win_rate,
            "minimum_color_win_rate": gate.minimum_color_win_rate,
            "maximum_errors": gate.maximum_errors,
            "maximum_move_ms": gate.maximum_move_ms,
            "minimum_ci95_low": gate.minimum_ci95_low,
            "maximum_color_gap": gate.maximum_color_gap,
            "maximum_p95_move_ms": gate.maximum_p95_move_ms,
        },
        "observed": {
            "games_completed": int(summary.get("games_completed", 0)),
            "ai_a_win_rate": float(summary.get("ai_a_win_rate", 0.0)),
            "ai_a_red_win_rate": float(summary.get("ai_a_red_win_rate", 0.0)),
            "ai_a_blue_win_rate": float(summary.get("ai_a_blue_win_rate", 0.0)),
            "errors": errors,
            "ai_a_max_move_ms": float(summary.get("ai_a_max_move_ms", 0.0)),
            "ai_a_win_rate_ci95_low": ci95_low,
            "color_gap": round(abs(red_rate - blue_rate), 6),
            "ai_a_p95_move_ms": p95_move_ms,
        },
    }


def promote_candidate(*, candidate_path, benchmark_path, models_dir,
                      selected_manifest_path=None, gate=None, apply=False):
    models_dir = Path(models_dir).resolve()
    candidate_path = Path(candidate_path).resolve()
    benchmark_path = Path(benchmark_path).resolve()
    selected_path = Path(
        selected_manifest_path or models_dir / "selected_model.json"
    ).resolve()
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    if not benchmark_path.is_file():
        raise FileNotFoundError(benchmark_path)
    validate_candidate_checkpoint(candidate_path)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    report = evaluate_promotion_gate(benchmark, gate)
    if not report["passed"]:
        return report
    candidate_hash = file_sha256(candidate_path)
    if apply and candidate_path.parent != models_dir:
        target = models_dir / candidate_path.name
        models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_path, target)
        source_config = candidate_path.with_name(
            f"{candidate_path.stem}_config.json"
        )
        source_sha = candidate_path.with_name(candidate_path.name + ".sha256")
        if source_config.is_file():
            shutil.copy2(source_config, target.with_name(f"{target.stem}_config.json"))
        if source_sha.is_file():
            shutil.copy2(source_sha, target.with_name(target.name + ".sha256"))
        candidate_path = target
    config_path = candidate_path.with_name(
        f"{candidate_path.stem}_config.json"
    )
    candidate_config = {}
    if config_path.is_file():
        candidate_config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f")
    backup_dir = models_dir / "backups"
    backup_manifest = backup_dir / f"selected_model_{timestamp}.json"
    history_path = models_dir / "selected_model_history.jsonl"
    previous_manifest = None
    if selected_path.is_file():
        previous_manifest = json.loads(selected_path.read_text(encoding="utf-8-sig"))
    if apply:
        backup_dir.mkdir(parents=True, exist_ok=True)
        if selected_path.is_file():
            shutil.copy2(selected_path, backup_manifest)
    manifest = {
        "schema_version": 1,
        "model_file": candidate_path.name,
        "model_version": candidate_config.get("model_version", "5.0.0"),
        "encoding_version": candidate_config.get("encoding_version", "3x11x11-v1"),
        "sha256": candidate_hash,
        "checkpoint_kind": candidate_config.get(
            "checkpoint_kind", "memory_supervised_candidate"
        ),
        "benchmark_file": str(benchmark_path),
        "training_samples": candidate_config.get("training_samples"),
        "dataset_manifest": candidate_config.get("dataset_manifest"),
        "dataset_manifest_sha256": candidate_config.get(
            "dataset_manifest_sha256"
        ),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    report["candidate"] = manifest
    report["backup_manifest"] = (
        str(backup_manifest) if backup_manifest.is_file() else None
    )
    report["applied"] = bool(apply)
    if apply:
        _atomic_json(selected_path, manifest)
        history_entry = {
            "created_at": manifest["created_at"],
            "model_file": candidate_path.name,
            "sha256": candidate_hash,
            "benchmark_file": str(benchmark_path),
            "previous_model_file": (
                previous_manifest.get("model_file") if previous_manifest else None
            ),
            "previous_sha256": (
                previous_manifest.get("sha256") if previous_manifest else None
            ),
        }
        with history_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(history_entry, ensure_ascii=False) + "\n")
    return report


def rollback_selected_model(*, models_dir, selected_manifest_path=None):
    models_dir = Path(models_dir).resolve()
    selected_path = Path(
        selected_manifest_path or models_dir / "selected_model.json"
    ).resolve()
    history_path = models_dir / "selected_model_history.jsonl"
    if not history_path.is_file():
        raise ModelPromotionError("No promotion history is available")
    lines = [line for line in history_path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        raise ModelPromotionError("Promotion history is empty")
    last = json.loads(lines[-1])
    previous_file = last.get("previous_model_file")
    previous_hash = last.get("previous_sha256")
    if not previous_file or not previous_hash:
        raise ModelPromotionError("The last promotion has no previous model")
    previous_path = models_dir / previous_file
    if not previous_path.is_file() or file_sha256(previous_path) != previous_hash:
        raise ModelPromotionError("Previous model file is missing or corrupted")
    manifest = {
        "schema_version": 1,
        "model_file": previous_file,
        "sha256": previous_hash,
        "checkpoint_kind": "rollback",
        "rollback_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    config_path = previous_path.with_name(f"{previous_path.stem}_config.json")
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        manifest["model_version"] = config.get("model_version", "5.0.0")
        manifest["encoding_version"] = config.get(
            "encoding_version", "3x11x11-v1"
        )
    _atomic_json(selected_path, manifest)
    history_path.write_text("\n".join(lines[:-1]) + ("\n" if lines[:-1] else ""),
                            encoding="utf-8")
    return manifest