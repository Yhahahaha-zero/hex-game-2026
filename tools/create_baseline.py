# -*- coding: utf-8 -*-
"""Create a verifiable safe baseline for the competition project.

The live SQLite database is copied with the SQLite backup API so WAL state is
handled consistently. Local credentials are never copied or printed.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import subprocess
import sys
import zipfile

TOOLS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.memory.schema import MEMORY_TABLES


SKIP_DIR_NAMES = {
    ".git", ".venv", "venv", "__pycache__", "build", "dist",
    "llm_cache", "analysis_logs", "logs",
}
SKIP_FILE_NAMES = {"hex_records.json"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_env_settings(path):
    """Read non-secret runtime switches without exposing credential values."""
    path = Path(path)
    values = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        if not separator:
            continue
        name = name.strip()
        if name.startswith("HEX_MEMORY_"):
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            values[name] = value
    return values


def _effective_settings(env_file):
    settings = read_env_settings(env_file)
    for name in (
        "HEX_MEMORY_CAPTURE", "HEX_MEMORY_PIPELINE", "HEX_MEMORY_RETRIEVAL",
        "HEX_MEMORY_AUTO_TRAIN", "HEX_MEMORY_AUTO_PROMOTE",
        "HEX_MEMORY_AUTO_TRAIN_V4", "HEX_MEMORY_AUTO_PROMOTE_V4",
    ):
        if name in os.environ:
            settings[name] = str(os.environ[name]).strip()
    return settings


def _git(code_root, *args):
    try:
        completed = subprocess.run(
            ["git", "-C", str(code_root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed if completed.returncode == 0 else None


def _git_metadata(code_root):
    commit = _git(code_root, "rev-parse", "HEAD")
    status = _git(code_root, "status", "--porcelain")
    tracked = _git(code_root, "ls-files", "--error-unmatch", ".env.local")
    ignored = _git(code_root, "check-ignore", "-q", ".env.local")
    return {
        "available": commit is not None,
        "commit": commit.stdout.strip() if commit else None,
        "working_tree_dirty": bool(status.stdout.strip()) if status else None,
        "env_local_tracked": tracked is not None,
        "env_local_ignored": ignored is not None,
    }


def backup_sqlite(source, destination):
    """Create and verify a transactionally consistent SQLite backup."""
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"SQLite database not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(str(source))) as source_connection:
        with closing(sqlite3.connect(str(destination))) as destination_connection:
            source_connection.backup(destination_connection)
    with closing(sqlite3.connect(str(destination))) as connection:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        table_counts = {}
        for table in MEMORY_TABLES:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            table_counts[table] = (
                int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
                if exists else 0
            )
    return {
        "source": str(source.resolve()),
        "backup": str(destination.resolve()),
        "sha256": file_sha256(destination),
        "size_bytes": destination.stat().st_size,
        "integrity_check": integrity,
        "table_counts": table_counts,
    }


def _source_archive(code_root, destination, *, excluded_root=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    excluded_root = Path(excluded_root).resolve() if excluded_root else None
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for item in sorted(Path(code_root).rglob("*")):
            if not item.is_file():
                continue
            relative = item.relative_to(code_root)
            if any(part in SKIP_DIR_NAMES for part in relative.parts):
                continue
            if item.name in SKIP_FILE_NAMES or item.suffix.lower() in SKIP_SUFFIXES:
                continue
            if item.name == ".env" or (item.name.startswith(".env.") and item.name != ".env.example"):
                continue
            if excluded_root:
                try:
                    item.resolve().relative_to(excluded_root)
                    continue
                except ValueError:
                    pass
            archive.write(item, relative.as_posix())
    return destination


def _copy_if_present(source, destination):
    source = Path(source)
    if not source.is_file():
        return None
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "source": str(source.resolve()),
        "backup": str(destination.resolve()),
        "sha256": file_sha256(destination),
        "size_bytes": destination.stat().st_size,
    }


def _package_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _unique_directory(path):
    path = Path(path)
    if not path.exists():
        return path
    index = 1
    while True:
        candidate = path.with_name(f"{path.name}_{index:02d}")
        if not candidate.exists():
            return candidate
        index += 1


def _atomic_json(path, document):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_summary(path, manifest):
    checks = manifest["checks"]
    lines = [
        "# Baseline Snapshot",
        "",
        f"- Created: {manifest['created_at']}",
        f"- Commit: {manifest['git'].get('commit') or '-'}",
        f"- Working tree dirty: {manifest['git'].get('working_tree_dirty')}",
        f"- Competition profile: {manifest['profile'].get('profile_name', '-')}",
        f"- Memory mode: {manifest['profile'].get('memory_mode', '-')}",
        f"- Database integrity: {manifest['database'].get('integrity_check', '-')}",
        "",
        "## Safety checks",
        "",
    ]
    for name, value in checks.items():
        lines.append(f"- {name}: {value}")
    lines.extend([
        "",
        "## Protected assets",
        "",
    ])
    for name, item in manifest["assets"].items():
        if item:
            lines.append(f"- {name}: `{item['backup']}`")
            lines.append(f"  SHA256: `{item['sha256']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", default=str(PROJECT_ROOT))
    parser.add_argument("--workspace-root", default=str(WORKSPACE_ROOT))
    parser.add_argument(
        "--output",
        default=str(WORKSPACE_ROOT / "evaluation" / "baseline"),
    )
    parser.add_argument("--db", default=str(WORKSPACE_ROOT / "artifacts" / "memory" / "hex_experience.sqlite3"))
    parser.add_argument("--model", default=str(WORKSPACE_ROOT / "models" / "hex_model_v5.pt"))
    parser.add_argument("--selected-model", default=str(WORKSPACE_ROOT / "models" / "selected_model.json"))
    parser.add_argument("--candidate", default=str(WORKSPACE_ROOT / "artifacts" / "memory" / "candidates" / "hex_model_v5_memory_candidate.pt"))
    parser.add_argument("--profile", default=str(PROJECT_ROOT / "competition_profile.json"))
    parser.add_argument("--training-summary", default=str(WORKSPACE_ROOT / "evaluation" / "memory" / "memory_training_summary.json"))
    parser.add_argument("--test-summary", default=str(WORKSPACE_ROOT / "evaluation" / "test_results" / "final_check.txt"))
    parser.add_argument("--no-source-archive", action="store_true")
    args = parser.parse_args(argv)

    code_root = Path(args.code_root).resolve()
    workspace_root = Path(args.workspace_root).resolve()
    output = _unique_directory(Path(args.output))
    output.mkdir(parents=True, exist_ok=False)

    profile_path = Path(args.profile)
    selected_path = Path(args.selected_model)
    if not profile_path.is_file():
        raise FileNotFoundError(f"competition profile not found: {profile_path}")
    if not selected_path.is_file():
        raise FileNotFoundError(f"selected model manifest not found: {selected_path}")
    profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    selected = json.loads(selected_path.read_text(encoding="utf-8-sig"))

    database = backup_sqlite(args.db, output / "hex_experience.sqlite3")
    model = _copy_if_present(args.model, output / "hex_model_v5.pt")
    selected_copy = _copy_if_present(selected_path, output / "selected_model.json")
    profile_copy = _copy_if_present(profile_path, output / "competition_profile.json")
    candidate = _copy_if_present(args.candidate, output / "hex_model_v5_memory_candidate.pt")
    training_summary = _copy_if_present(args.training_summary, output / "memory_training_summary.json")
    test_summary = _copy_if_present(args.test_summary, output / "final_check.txt")

    source_archive = None
    if not args.no_source_archive:
        archive_path = _source_archive(
            code_root,
            output / "source_snapshot.zip",
            excluded_root=output,
        )
        source_archive = {
            "source": str(code_root),
            "backup": str(archive_path.resolve()),
            "sha256": file_sha256(archive_path),
            "size_bytes": archive_path.stat().st_size,
        }

    env_file = code_root / ".env.local"
    env_settings = _effective_settings(env_file)
    git = _git_metadata(code_root)
    expected_model_sha = str(selected.get("sha256", "")).lower()
    actual_model_sha = model["sha256"] if model else None
    flags_off = {"0", "false", "no", "off", ""}
    checks = {
        "profile_memory_shadow": str(profile.get("memory_mode", "")).lower() == "shadow",
        "auto_train_disabled": env_settings.get("HEX_MEMORY_AUTO_TRAIN", "0").lower() in flags_off
        and env_settings.get("HEX_MEMORY_AUTO_TRAIN_V4", "0").lower() in flags_off,
        "auto_promotion_disabled": env_settings.get("HEX_MEMORY_AUTO_PROMOTE", "0").lower() in flags_off
        and env_settings.get("HEX_MEMORY_AUTO_PROMOTE_V4", "0").lower() in flags_off,
        "selected_model_sha256_matches": bool(actual_model_sha and actual_model_sha == expected_model_sha),
        "database_integrity_ok": database["integrity_check"] == "ok",
        "env_local_not_copied": not any(
            path.name == ".env.local" for path in output.rglob("*") if path.is_file()
        ),
    }
    if env_file.is_file():
        checks["env_local_not_tracked"] = git["env_local_tracked"] is False
        checks["env_local_ignored"] = git["env_local_ignored"] is True

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "workspace_root": str(workspace_root),
        "code_root": str(code_root),
        "output": str(output.resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "pygame": _package_version("pygame"),
            "torch": _package_version("torch"),
            "numpy": _package_version("numpy"),
            "Pillow": _package_version("Pillow"),
        },
        "git": git,
        "profile": {
            "profile_name": profile.get("profile_name"),
            "profile_version": profile.get("profile_version"),
            "ai_engine": profile.get("ai_engine"),
            "memory_mode": profile.get("memory_mode"),
            "memory_db": profile.get("memory_db"),
            "random_seed": profile.get("random_seed"),
        },
        "memory_settings": {
            "capture": env_settings.get("HEX_MEMORY_CAPTURE"),
            "pipeline": env_settings.get("HEX_MEMORY_PIPELINE"),
            "retrieval": env_settings.get("HEX_MEMORY_RETRIEVAL"),
            "auto_train": env_settings.get("HEX_MEMORY_AUTO_TRAIN", "0"),
            "auto_promote": env_settings.get("HEX_MEMORY_AUTO_PROMOTE", "0"),
            "auto_train_v4": env_settings.get("HEX_MEMORY_AUTO_TRAIN_V4", "0"),
            "auto_promote_v4": env_settings.get("HEX_MEMORY_AUTO_PROMOTE_V4", "0"),
        },
        "database": database,
        "selected_model": selected,
        "assets": {
            "model": model,
            "selected_model": selected_copy,
            "competition_profile": profile_copy,
            "candidate": candidate,
            "training_summary": training_summary,
            "test_summary": test_summary,
            "source_archive": source_archive,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    _atomic_json(output / "manifest.json", manifest)
    _write_summary(output / "BASELINE.md", manifest)
    print(json.dumps({
        "output": str(output),
        "passed": manifest["passed"],
        "checks": checks,
        "database": database["table_counts"],
        "model_sha256": actual_model_sha,
    }, ensure_ascii=False, indent=2))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
