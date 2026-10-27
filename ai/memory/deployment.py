# -*- coding: utf-8 -*-
"""Atomic deployment manifest for model and retrieval-policy bundles."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


DEPLOYMENT_SCHEMA_VERSION = 1


class DeploymentError(RuntimeError):
    pass


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


def build_deployment_manifest(*, models_dir, model_path, policy_path,
                              benchmark_file=None, dataset_id=None,
                              feature_version=1, rules_version=1):
    models_dir = Path(models_dir).resolve()
    model_path = Path(model_path).resolve()
    policy_path = Path(policy_path).resolve()
    if not model_path.is_file() or not policy_path.is_file():
        raise FileNotFoundError("model and retrieval policy are required")
    checkpoint = _load_checkpoint(model_path)
    return {
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "model_file": model_path.name,
        "model_sha256": file_sha256(model_path),
        "model_version": checkpoint["model_version"],
        "encoding_version": checkpoint["encoding_version"],
        "memory_policy_file": policy_path.name,
        "memory_policy_sha256": file_sha256(policy_path),
        "feature_version": int(feature_version),
        "rules_version": int(rules_version),
        "benchmark_file": str(benchmark_file) if benchmark_file else None,
        "dataset_id": dataset_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def _load_checkpoint(path):
    try:
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise DeploymentError(f"model checkpoint cannot be loaded: {exc}") from exc
    if not isinstance(checkpoint, dict) or not checkpoint.get("model_state_dict"):
        raise DeploymentError("deployment model is not a structured checkpoint")
    if checkpoint.get("model_name") != "HexNetV5":
        raise DeploymentError("deployment model_name is incompatible")
    if checkpoint.get("model_version") != "5.0.0":
        raise DeploymentError("deployment model_version is incompatible")
    if checkpoint.get("encoding_version") != "3x11x11-v1":
        raise DeploymentError("deployment encoding_version is incompatible")
    return checkpoint


def _validate_checkpoint(path):
    _load_checkpoint(path)


def validate_deployment_manifest(manifest_path, *, models_dir=None):
    manifest_path = Path(manifest_path).resolve()
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(document.get("schema_version", 0)) != DEPLOYMENT_SCHEMA_VERSION:
        raise DeploymentError("unsupported deployment schema")
    root = Path(models_dir).resolve() if models_dir else manifest_path.parent
    validate_deployment_manifest_from_document(document, root)
    return document


def activate_deployment(*, models_dir, manifest, apply=False):
    models_dir = Path(models_dir).resolve()
    target = models_dir / "deployment_manifest.json"
    validate_deployment_manifest_from_document(manifest, models_dir)
    if not apply:
        return {"applied": False, "manifest": manifest, "backup": None}
    backup = models_dir / "deployment_manifest.previous.json"
    if target.is_file():
        descriptor, temp_name = tempfile.mkstemp(
            dir=models_dir, prefix=".deployment.previous.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(target.read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, backup)
        finally:
            if Path(temp_name).exists():
                Path(temp_name).unlink()
    _atomic_json(target, manifest)
    return {
        "applied": True,
        "manifest": manifest,
        "backup": str(backup) if backup.is_file() else None,
    }


def validate_deployment_manifest_from_document(document, models_dir):
    if not isinstance(document, dict):
        raise DeploymentError("deployment manifest must be a JSON object")
    if int(document.get("schema_version", 0)) != DEPLOYMENT_SCHEMA_VERSION:
        raise DeploymentError("unsupported deployment schema")
    if document.get("feature_version") != 1 or document.get("rules_version") != 1:
        raise DeploymentError("deployment version mismatch")
    for key in ("model_file", "memory_policy_file"):
        relative = Path(str(document.get(key, "")))
        if not relative.name or relative.is_absolute() or ".." in relative.parts:
            raise DeploymentError(f"invalid {key}")
    for key in ("model_sha256", "memory_policy_sha256"):
        value = document.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
            raise DeploymentError(f"invalid {key}")
    model_path = Path(models_dir) / document["model_file"]
    policy_path = Path(models_dir) / document["memory_policy_file"]
    if not model_path.is_file() or file_sha256(model_path) != document["model_sha256"]:
        raise DeploymentError("model hash mismatch")
    _validate_checkpoint(model_path)
    if not policy_path.is_file() or file_sha256(policy_path) != document["memory_policy_sha256"]:
        raise DeploymentError("memory policy hash mismatch")
    selected_path = Path(models_dir) / "selected_model.json"
    if selected_path.is_file():
        try:
            selected = json.loads(selected_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise DeploymentError("selected_model.json cannot be parsed") from exc
        if not isinstance(selected, dict):
            raise DeploymentError("selected_model.json must contain a JSON object")
        if selected.get("model_file") != document.get("model_file"):
            raise DeploymentError("deployment model does not match selected_model.json")
        if selected.get("sha256") != document.get("model_sha256"):
            raise DeploymentError("deployment model hash does not match selected_model.json")
    return True


def rollback_deployment(*, models_dir):
    models_dir = Path(models_dir).resolve()
    target = models_dir / "deployment_manifest.json"
    backup = models_dir / "deployment_manifest.previous.json"
    if not backup.is_file():
        raise DeploymentError("no previous deployment manifest is available")
    document = validate_deployment_manifest(backup, models_dir=models_dir)
    _atomic_json(target, document)
    return document