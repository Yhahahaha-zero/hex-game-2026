# -*- coding: utf-8 -*-
"""Build, validate, activate or roll back a V4 model+policy deployment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.memory.deployment import (
    activate_deployment,
    build_deployment_manifest,
    rollback_deployment,
    validate_deployment_manifest,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT.parent / "models"))
    parser.add_argument("--model")
    parser.add_argument("--policy")
    parser.add_argument("--benchmark")
    parser.add_argument("--dataset-id")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args(argv)
    models_dir = Path(args.models_dir)
    if args.rollback:
        print(json.dumps(rollback_deployment(models_dir=models_dir), ensure_ascii=False, indent=2))
        return 0
    if not args.model or not args.policy:
        raise ValueError("--model and --policy are required")
    manifest = build_deployment_manifest(
        models_dir=models_dir,
        model_path=args.model,
        policy_path=args.policy,
        benchmark_file=args.benchmark,
        dataset_id=args.dataset_id,
    )
    result = activate_deployment(
        models_dir=models_dir,
        manifest=manifest,
        apply=args.apply,
    )
    if args.apply:
        validate_deployment_manifest(
            models_dir / "deployment_manifest.json",
            models_dir=models_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())