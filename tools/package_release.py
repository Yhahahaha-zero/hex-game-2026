# -*- coding: utf-8 -*-
"""Create reproducible development, demo and evaluation archive packages."""
import argparse
from datetime import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.neural_ai import load_selected_model

DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "dist"
VALID_PROFILES = ("dev", "demo", "archive")
EXCLUDED_DIRS = {".git", "__pycache__", ".venv", "venv", "tmp", "dist", "artifacts"}


def _is_excluded(path):
    return (
        any(part in EXCLUDED_DIRS for part in path.parts)
        or path.suffix in {".pyc", ".pyo", ".tmp"}
    )


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add(entries, workspace_root, path):
    path = Path(path)
    if not path.is_file() or _is_excluded(path.relative_to(workspace_root)):
        return
    relative = path.relative_to(workspace_root).as_posix()
    entries[relative] = path


def _collect_tree(entries, workspace_root, root):
    root = Path(root)
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file():
            _add(entries, workspace_root, path)


def _selected_assets(workspace_root):
    models_root = workspace_root / "models"
    manifest = models_root / "selected_model.json"
    model_path, metadata = load_selected_model(manifest)
    assets = [manifest, model_path]
    sidecar = model_path.with_name(model_path.name + ".sha256")
    config = model_path.with_name(model_path.stem + "_config.json")
    common_config = models_root / "model_config.json"
    for path in (sidecar, config, common_config):
        if path.is_file():
            assets.append(path)
    return assets, metadata


def _demo_code_files(workspace_root):
    code_root = workspace_root / "hex_game"
    root_names = {
        "__init__.py", "hex_game.py", "hex_board.py", "hex_engine.py",
        "hex_ai.py", "game_mode.py", "game_timer.py", "game_records.py",
        "report_export.py", "hex_notation.py", "start_game.bat",
        "competition_profile.json",
        "requirements.txt", "requirements-ai.txt", "requirements-dev.txt",
        "pyproject.toml",
    }
    ai_names = {
        "__init__.py", "explanation.py", "evaluation.py", "tactics.py",
        "encode.py", "hybrid_ai.py", "strategic_ai.py", "neural_ai.py",
        "ai_factory.py", "mcts.py", "model_v5.py", "search_result.py",
        "worker.py",
    }
    files = []
    for name in root_names:
        path = code_root / name
        if path.is_file():
            files.append(path)
    for name in ai_names:
        path = code_root / "ai" / name
        if path.is_file():
            files.append(path)
    for folder in ("ui",):
        files.extend(path for path in (code_root / folder).glob("*.py") if path.is_file())
    memory_root = code_root / "ai" / "memory"
    files.extend(
        path for path in memory_root.glob("*.py") if path.is_file()
    )
    for name in ("__init__.py", "check_model.py"):
        path = code_root / "tools" / name
        if path.is_file():
            files.append(path)
    return files


def collect_release_files(profile, workspace_root=WORKSPACE_ROOT):
    """Return mapping archive-path -> source-path for one profile."""
    workspace_root = Path(workspace_root).resolve()
    code_root = workspace_root / "hex_game"
    entries = {}

    if profile not in VALID_PROFILES:
        raise ValueError(f"未知发布类型: {profile!r}")

    if profile == "dev":
        _collect_tree(entries, workspace_root, code_root)
        _collect_tree(entries, workspace_root, workspace_root / "models")
        _collect_tree(entries, workspace_root, workspace_root / "evaluation")
        _collect_tree(entries, workspace_root, workspace_root / "docs")
        _add(entries, workspace_root, workspace_root / "README.md")
    elif profile == "demo":
        for path in _demo_code_files(workspace_root):
            _add(entries, workspace_root, path)
        selected_files, _ = _selected_assets(workspace_root)
        for path in selected_files:
            _add(entries, workspace_root, path)
        for name in (
            "README.md", "docs/COMPETITION_PROFILE.md", "docs/MODEL_AND_CHECKPOINTS.md", "docs/HEX_NOTATION.md",
            "docs/RECORDS.md", "docs/REPORTS.md", "docs/MEMORY_INTERFACE.md",
            "docs/MEMORY_V1.md", "docs/MEMORY_V2.md", "docs/MEMORY_V3.md",
            "docs/MEMORY_V4.md", "hex_game/.env.example",
        ):
            _add(entries, workspace_root, workspace_root / name)
    elif profile == "archive":
        selected_files, _ = _selected_assets(workspace_root)
        for path in selected_files:
            _add(entries, workspace_root, path)
        _collect_tree(entries, workspace_root, workspace_root / "evaluation")
        _add(
            entries, workspace_root,
            workspace_root / "hex_game" / "competition_profile.json",
        )
        for name in (
            "README.md", "docs/COMPETITION_PROFILE.md", "docs/SYSTEM_ARCHITECTURE.png",
            "docs/SYSTEM_ARCHITECTURE.mmd", "docs/SYSTEM_ARCHITECTURE.md",
            "docs/MODEL_AND_CHECKPOINTS.md", "docs/AI_EVALUATION.md",
            "docs/P1_EVALUATION.md", "docs/TESTING.md",
            "docs/MEMORY_INTERFACE.md", "docs/MEMORY_V1.md",
            "docs/MEMORY_V2.md", "docs/MEMORY_V3.md", "docs/MEMORY_V4.md",
        ):
            _add(entries, workspace_root, workspace_root / name)

    return entries


def _git_commit(workspace_root):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace_root / "hex_game",
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def environment_info(workspace_root=WORKSPACE_ROOT):
    workspace_root = Path(workspace_root).resolve()
    packages = {}
    for name in ("pygame", "torch", "numpy"):
        try:
            module = importlib.import_module(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            packages[name] = None
    model_path, metadata = load_selected_model(
        workspace_root / "models" / "selected_model.json"
    )
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "git_commit": _git_commit(workspace_root),
        "packages": packages,
        "selected_model": metadata,
        "model_path": model_path.relative_to(workspace_root).as_posix(),
        "model_sha256": _file_sha256(model_path),
    }


def build_release_archive(profile, output_dir=DEFAULT_OUTPUT_DIR, name=None,
                          workspace_root=WORKSPACE_ROOT):
    workspace_root = Path(workspace_root).resolve()
    entries = collect_release_files(profile, workspace_root)
    if not entries:
        raise ValueError(f"{profile} 没有可打包文件")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = name or f"hex-game-2026-{profile}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    archive_path = output_dir / f"{stem}.zip"
    prefix = stem
    manifest = {
        "profile": profile,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "file_count": len(entries),
        "files": [
            {"path": relative, "size": source.stat().st_size}
            for relative, source in sorted(entries.items())
        ],
        "environment": environment_info(workspace_root),
    }
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for relative, source in sorted(entries.items()):
            zf.write(source, f"{prefix}/{relative}")
        zf.writestr(
            f"{prefix}/release_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
        zf.writestr(
            f"{prefix}/ENVIRONMENT.json",
            json.dumps(manifest["environment"], ensure_ascii=False, indent=2),
        )
    return archive_path, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Package Hex project releases")
    parser.add_argument("--profile", choices=(*VALID_PROFILES, "all"), default="demo")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)
    profiles = VALID_PROFILES if args.profile == "all" else (args.profile,)
    created = []
    try:
        for profile in profiles:
            name = args.name
            if name and len(profiles) > 1:
                name = f"{name}-{profile}"
            path, manifest = build_release_archive(
                profile, output_dir=args.output_dir, name=name
            )
            created.append(path)
            print(f"[OK] {profile}: {path} ({manifest['file_count']} files)")
    except Exception as exc:
        print(f"[ERROR] package failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
