# -*- coding: utf-8 -*-
"""Verify an immutable competition baseline and all referenced hashes."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.competition_config import file_sha256
from tools import verify_competition_results


BASELINE_SCHEMA_VERSION = 1


def _add_check(checks, errors, name, condition, message):
    checks[name] = bool(condition)
    if not condition:
        errors.append(message)


def _safe_relative(root, value):
    path = Path(str(value))
    if not path.name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe baseline path: {value!r}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"baseline path escapes root: {value!r}") from exc
    return resolved


def verify_baseline(baseline_dir, *, max_p99_ms=None):
    root = Path(baseline_dir).resolve()
    manifest_path = root / "baseline_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("baseline_manifest.json is missing")
    document = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    checks = {}
    errors = []
    _add_check(
        checks, errors, "schema",
        int(document.get("baseline_schema_version", 0)) == BASELINE_SCHEMA_VERSION,
        "unsupported baseline schema",
    )
    commit = str(document.get("git_commit", ""))
    _add_check(
        checks, errors, "commit",
        bool(re.fullmatch(r"[0-9a-fA-F]{7,64}", commit)),
        "baseline git_commit is missing or invalid",
    )
    _add_check(
        checks, errors, "clean_source", document.get("working_tree_dirty") is False,
        "baseline source worktree was dirty",
    )
    file_records = document.get("files", [])
    seen = set()
    files_valid = bool(file_records)
    roles = {}
    for record in file_records:
        relative = str(record.get("path", ""))
        if relative in seen:
            files_valid = False
            continue
        seen.add(relative)
        try:
            path = _safe_relative(root, relative)
            if (not path.is_file()
                    or path.stat().st_size != int(record.get("size", -1))
                    or file_sha256(path) != record.get("sha256")):
                files_valid = False
            roles[record.get("role")] = path
        except (OSError, ValueError, TypeError):
            files_valid = False
    _add_check(checks, errors, "file_hashes", files_valid,
               "one or more baseline files are missing, changed or unsafe")
    for role in ("competition_profile", "tournament_result", "test_report",
                 "selected_model_manifest"):
        _add_check(
            checks, errors, f"required_{role}", role in roles,
            f"required baseline role is missing: {role}",
        )

    profile_path = roles.get("competition_profile")
    frozen = None
    if profile_path and profile_path.is_file():
        frozen = json.loads(profile_path.read_text(encoding="utf-8-sig"))
        profile_meta = document.get("frozen_profile", {})
        _add_check(
            checks, errors, "frozen_profile_hash",
            profile_meta.get("sha256") == file_sha256(profile_path),
            "frozen profile hash mismatch",
        )
        _add_check(
            checks, errors, "frozen_profile_commit",
            frozen.get("freeze_commit") == commit,
            "frozen profile commit does not match baseline commit",
        )
        _add_check(
            checks, errors, "frozen_profile_engine",
            frozen.get("ai_engine") == profile_meta.get("ai_engine"),
            "frozen profile AI engine mismatch",
        )
        _add_check(
            checks, errors, "frozen_profile_memory",
            frozen.get("memory_mode") == profile_meta.get("memory_mode"),
            "frozen profile memory mode mismatch",
        )

    model = document.get("model") or {}
    model_ok = True
    if model.get("file"):
        if model.get("copied"):
            model_path = roles.get("model_checkpoint")
            model_ok = bool(model_path and model_path.is_file())
        else:
            source_path = Path(str(model.get("source_path", "")))
            model_path = source_path if source_path.is_absolute() else WORKSPACE_ROOT / source_path
            model_ok = model_path.is_file()
        if model_ok:
            model_ok = file_sha256(model_path) == model.get("sha256")
    _add_check(checks, errors, "model_hash", model_ok,
               "model file is missing or does not match the baseline hash")

    selected_path = roles.get("selected_model_manifest")
    if selected_path and selected_path.is_file() and model.get("file"):
        selected = json.loads(selected_path.read_text(encoding="utf-8-sig"))
        _add_check(
            checks, errors, "selected_model_matches",
            selected.get("model_file") == model.get("file")
            and selected.get("sha256") == model.get("sha256"),
            "selected_model.json does not match the frozen model",
        )

    tests = document.get("tests", {})
    _add_check(
        checks, errors, "test_count",
        int(tests.get("test_count", 0)) > 0,
        "baseline test count is missing or zero",
    )
    _add_check(
        checks, errors, "test_result", tests.get("test_result") == "OK",
        "baseline tests are not marked OK",
    )
    test_path = roles.get("test_report")
    if test_path and test_path.is_file():
        test_text = test_path.read_text(encoding="utf-8-sig")
        _add_check(
            checks, errors, "test_report_ok",
            "test_result=OK" in test_text and f"test_count={tests.get('test_count')}" in test_text,
            "copied final_check.txt is inconsistent with the manifest",
        )

    source_profile = document.get("source_profile", {})
    source_profile_sha = source_profile.get("sha256")
    _add_check(
        checks, errors, "source_profile_hash",
        isinstance(source_profile_sha, str) and len(source_profile_sha) == 64,
        "source profile SHA256 is missing or invalid",
    )
    hard_limit = float(
        (frozen or {}).get(
            "hard_move_limit_ms",
            (document.get("frozen_profile") or {}).get("hard_move_limit_ms", 500),
        ) or 500
    )
    limit = hard_limit if max_p99_ms is None else float(max_p99_ms)
    validations = {}
    for role, kind, require_recommendation in (
        ("tournament_result", "tournament", True),
        ("memory_ablation_result", "memory_ablation", False),
        ("benchmark_result", "benchmark", False),
    ):
        path = roles.get(role)
        if not path or not path.is_file():
            if role == "tournament_result":
                _add_check(checks, errors, f"{kind}_validation", False,
                           "tournament result is missing")
            continue
        try:
            result_document = json.loads(path.read_text(encoding="utf-8-sig"))
            result = verify_competition_results.verify_document(
                result_document,
                max_p99_ms=limit,
                allow_dirty=False,
                allow_timeouts=False,
                allow_unverified_db=False,
                require_recommendation=require_recommendation,
                expected_profile_sha256=source_profile_sha,
                require_minimum_pairs=True,
            )
            validations[kind] = result
            _add_check(
                checks, errors, f"{kind}_validation",
                result["passed"], f"{kind} validation failed",
            )
        except (OSError, ValueError, TypeError) as exc:
            validations[kind] = {"passed": False, "errors": [str(exc)]}
            _add_check(
                checks, errors, f"{kind}_validation", False,
                f"{kind} cannot be validated: {exc}",
            )
    return {
        "verification_schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "baseline_dir": str(root),
        "baseline_id": document.get("baseline_id"),
        "git_commit": commit,
        "passed": not errors,
        "checks": checks,
        "errors": errors,
        "result_validations": validations,
        "manifest_sha256": file_sha256(manifest_path),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify a frozen competition baseline")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--max-p99-ms", type=float, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    try:
        report = verify_baseline(
            args.baseline_dir, max_p99_ms=args.max_p99_ms
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"[ERROR] {exc}")
        return 2
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())