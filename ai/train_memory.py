# -*- coding: utf-8 -*-
"""Offline policy/value training from validated memory datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from ai.memory.dataset import load_memory_dataset
from ai.model_v5 import ENCODING_VERSION, MODEL_NAME, MODEL_VERSION, HexNetV5
from ai.train_v5 import save_checkpoint, set_training_seed

MODEL_DIR = PROJECT_ROOT.parent / "models"
DEFAULT_OUTPUT = MODEL_DIR / "hex_model_v5_memory_candidate.pt"
DEFAULT_SUMMARY = (
    PROJECT_ROOT.parent / "evaluation" / "training" / "memory_training_summary.json"
)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_parent_state(model, path):
    if path is None or str(path).lower() in ("", "none"):
        return None
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    return _file_sha256(path)


def _subset(arrays, split_code):
    mask = arrays["split_codes"] == split_code
    return {
        "states": torch.from_numpy(arrays["states"][mask]),
        "legal_masks": torch.from_numpy(arrays["legal_masks"][mask]),
        "actions": torch.from_numpy(arrays["actions"][mask]),
        "value_targets": torch.from_numpy(arrays["value_targets"][mask]),
        "policy_targets": torch.from_numpy(arrays["policy_targets"][mask]),
    }


def _loss(model, batch):
    policy_logits, values = model(batch["states"])
    masked_logits = policy_logits.masked_fill(~batch["legal_masks"], -1e9)
    log_probabilities = F.log_softmax(masked_logits, dim=1)
    policy_loss = -(
        batch["policy_targets"] * log_probabilities
    ).sum(dim=1).mean()
    value_loss = F.mse_loss(values, batch["value_targets"])
    return policy_loss, value_loss


def _evaluate(model, batch):
    model.eval()
    with torch.no_grad():
        policy_loss, value_loss = _loss(model, batch)
    return float(policy_loss.item()), float(value_loss.item())


def _train_epoch(model, optimizer, batch, *, batch_size, policy_weight, value_weight):
    model.train()
    sample_count = int(batch["states"].shape[0])
    indices = torch.randperm(sample_count)
    policy_total = 0.0
    value_total = 0.0
    for start in range(0, sample_count, batch_size):
        selected = indices[start:start + batch_size]
        chunk = {name: value[selected] for name, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        policy_loss, value_loss = _loss(model, chunk)
        loss = policy_weight * policy_loss + value_weight * value_loss
        loss.backward()
        optimizer.step()
        policy_total += float(policy_loss.item()) * len(selected)
        value_total += float(value_loss.item()) * len(selected)
    return policy_total / sample_count, value_total / sample_count


def train_memory_model(*, dataset_dir, output_path=DEFAULT_OUTPUT,
                       parent_checkpoint=None, epochs=5, batch_size=32,
                       learning_rate=1e-4, policy_weight=1.0,
                       value_weight=0.5, seed=20261025, device="cpu",
                       summary_path=DEFAULT_SUMMARY):
    set_training_seed(seed)
    arrays, manifest = load_memory_dataset(dataset_dir)
    train_batch = _subset(arrays, 0)
    validation_batch = _subset(arrays, 1)
    test_batch = _subset(arrays, 2)
    if train_batch["states"].shape[0] == 0:
        raise ValueError("dataset has no training samples")
    if validation_batch["states"].shape[0] == 0:
        validation_batch = train_batch
    if int(epochs) < 1 or int(batch_size) < 1:
        raise ValueError("epochs and batch_size must be positive")
    device = str(device)
    model = HexNetV5().to(device)
    parent_sha256 = _load_parent_state(model, parent_checkpoint)
    train_batch = {name: value.to(device) for name, value in train_batch.items()}
    validation_batch = {
        name: value.to(device) for name, value in validation_batch.items()
    }
    test_batch = {name: value.to(device) for name, value in test_batch.items()}
    optimizer = Adam(model.parameters(), lr=float(learning_rate))
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(1, int(epochs) + 1):
        policy_loss, value_loss = _train_epoch(
            model,
            optimizer,
            train_batch,
            batch_size=int(batch_size),
            policy_weight=float(policy_weight),
            value_weight=float(value_weight),
        )
        validation_policy, validation_value = _evaluate(model, validation_batch)
        validation_total = (
            float(policy_weight) * validation_policy
            + float(value_weight) * validation_value
        )
        if validation_total < best_loss:
            best_loss = validation_total
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        history.append({
            "epoch": epoch,
            "policy_loss": round(policy_loss, 8),
            "value_loss": round(value_loss, 8),
            "validation_policy_loss": round(validation_policy, 8),
            "validation_value_loss": round(validation_value, 8),
            "validation_total_loss": round(validation_total, 8),
        })
    if best_state is not None:
        model.load_state_dict(best_state)
    test_policy, test_value = (
        _evaluate(model, test_batch)
        if test_batch["states"].shape[0] else (None, None)
    )
    manifest_path = Path(dataset_dir) / "manifest.json"
    metadata = {
        "checkpoint_kind": "memory_supervised_candidate",
        "training_games": int(len(manifest["match_ids"])),
        "training_samples": int(manifest["sample_count"]),
        "dataset_manifest": str(manifest_path.resolve()),
        "dataset_manifest_sha256": _file_sha256(manifest_path),
        "dataset_samples_sha256": manifest["samples_sha256"],
        "dataset_split_counts": manifest["split_counts"],
        "dataset_seed": manifest["seed"],
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "policy_weight": float(policy_weight),
        "value_weight": float(value_weight),
        "pytorch_version": str(torch.__version__),
        "device": device,
        "parent_model_sha256": parent_sha256,
        "validation_loss": round(best_loss, 8),
        "test_policy_loss": round(test_policy, 8) if test_policy is not None else None,
        "test_value_loss": round(test_value, 8) if test_value is not None else None,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "history": history,
    }
    digest = save_checkpoint(model, Path(output_path), metadata)
    summary = {
        "schema_version": 1,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "encoding_version": ENCODING_VERSION,
        "checkpoint": str(Path(output_path).resolve()),
        "sha256": digest,
        **metadata,
    }
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--parent-checkpoint", default="none")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    summary = train_memory_model(
        dataset_dir=args.dataset,
        output_path=args.output,
        parent_checkpoint=args.parent_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        policy_weight=args.policy_weight,
        value_weight=args.value_weight,
        seed=args.seed,
        device=args.device,
        summary_path=args.summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())