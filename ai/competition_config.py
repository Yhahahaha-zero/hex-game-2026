# -*- coding: utf-8 -*-
"""Single source of truth for the competition AI configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_PROFILE_PATH = PROJECT_ROOT / "competition_profile.json"
MODELS_ROOT = WORKSPACE_ROOT / "models"
PROFILE_SCHEMA_VERSION = 1
ALLOWED_AI_ENGINES = (
    "basic", "strategic", "hybrid", "hybrid_llm", "neural"
)
ALLOWED_MEMORY_MODES = ("off", "shadow", "assist")


class CompetitionConfigError(ValueError):
    """Raised when the formal competition profile is missing or inconsistent."""


@dataclass(frozen=True)
class CompetitionSettings:
    profile_name: str
    profile_version: str
    ai_engine: str
    search_time_limit_ms: float
    hard_move_limit_ms: float
    random_seed: int
    memory_mode: str
    tournament_games: int
    tournament_final_games: int
    memory_db: str | None = None
    neural_model_file: str | None = None
    neural_model_sha256: str | None = None
    benchmark_file: str | None = None
    freeze_commit: str | None = None
    notes: str = ""
    profile_path: str = ""
    profile_sha256: str = ""

    @property
    def neural_model_path(self):
        if not self.neural_model_file:
            return None
        return MODELS_ROOT / self.neural_model_file

    def as_dict(self):
        return asdict(self)

    def metadata(self):
        return {
            "profile_name": self.profile_name,
            "profile_version": self.profile_version,
            "profile_path": self.profile_path,
            "profile_sha256": self.profile_sha256,
            "ai_engine": self.ai_engine,
            "search_time_limit_ms": self.search_time_limit_ms,
            "hard_move_limit_ms": self.hard_move_limit_ms,
            "random_seed": self.random_seed,
            "memory_mode": self.memory_mode,
            "neural_model_file": self.neural_model_file,
            "neural_model_sha256": self.neural_model_sha256,
            "freeze_commit": self.freeze_commit,
        }


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_text(document, key, *, optional=False):
    value = document.get(key)
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CompetitionConfigError(f"{key} must be a non-empty string")
    return value.strip()


def _require_int(document, key, *, minimum=0, maximum=None):
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompetitionConfigError(f"{key} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise CompetitionConfigError(f"{key} is outside the allowed range")
    return value


def _require_even_positive(document, key):
    value = _require_int(document, key, minimum=2)
    if value % 2:
        raise CompetitionConfigError(f"{key} must be an even integer")
    return value


def _validate_model(document, settings):
    model_file = settings.neural_model_file
    model_sha = settings.neural_model_sha256
    if model_file is None and model_sha is None:
        if settings.ai_engine == "neural":
            raise CompetitionConfigError(
                "neural engine requires neural_model_file and neural_model_sha256"
            )
        return
    if model_file is None or model_sha is None:
        raise CompetitionConfigError("model file and SHA256 must be provided together")
    model_path = Path(model_file)
    if model_path.name != model_file or model_path.is_absolute():
        raise CompetitionConfigError("neural_model_file must be a file name only")
    if re.fullmatch(r"[0-9a-fA-F]{64}", model_sha) is None:
        raise CompetitionConfigError("neural_model_sha256 must be a SHA256 hex digest")
    resolved = MODELS_ROOT / model_file
    if not resolved.is_file():
        raise CompetitionConfigError(f"model file does not exist: {resolved}")
    actual = file_sha256(resolved)
    if actual.lower() != model_sha.lower():
        raise CompetitionConfigError("neural model SHA256 mismatch")
    selected_path = MODELS_ROOT / "selected_model.json"
    if selected_path.is_file():
        try:
            selected = json.loads(selected_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise CompetitionConfigError("selected_model.json cannot be parsed") from exc
        if selected.get("model_file") != model_file:
            raise CompetitionConfigError(
                "competition profile model file does not match selected_model.json"
            )
        if str(selected.get("sha256", "")).lower() != model_sha.lower():
            raise CompetitionConfigError(
                "competition profile model does not match selected_model.json"
            )


def load_competition_settings(path=None, *, environ=None, verify_model=True):
    environment = os.environ if environ is None else environ
    configured = path or environment.get("HEX_COMPETITION_PROFILE") or DEFAULT_PROFILE_PATH
    profile_path = Path(configured).expanduser().resolve()
    if not profile_path.is_file():
        raise CompetitionConfigError(f"competition profile not found: {profile_path}")
    try:
        document = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise CompetitionConfigError("competition profile cannot be parsed") from exc
    if not isinstance(document, dict):
        raise CompetitionConfigError("competition profile must contain a JSON object")
    if document.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise CompetitionConfigError("unsupported competition profile schema")

    ai_engine = _require_text(document, "ai_engine").lower()
    if ai_engine not in ALLOWED_AI_ENGINES:
        raise CompetitionConfigError(f"unsupported ai_engine: {ai_engine!r}")
    memory_mode = str(document.get("memory_mode", "off")).strip().lower()
    if memory_mode not in ALLOWED_MEMORY_MODES:
        raise CompetitionConfigError(f"unsupported memory_mode: {memory_mode!r}")
    search_time_limit = document.get("search_time_limit_ms")
    hard_move_limit = document.get("hard_move_limit_ms")
    for name, value in (
        ("search_time_limit_ms", search_time_limit),
        ("hard_move_limit_ms", hard_move_limit),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CompetitionConfigError(f"{name} must be a number")
    search_time_limit = float(search_time_limit)
    hard_move_limit = float(hard_move_limit)
    if search_time_limit < 80.0 or search_time_limit > 5000.0:
        raise CompetitionConfigError("search_time_limit_ms must be between 80 and 5000")
    if hard_move_limit < search_time_limit or hard_move_limit > 10000.0:
        raise CompetitionConfigError(
            "hard_move_limit_ms must be >= search time and <= 10000"
        )

    settings = CompetitionSettings(
        profile_name=_require_text(document, "profile_name"),
        profile_version=_require_text(document, "profile_version"),
        ai_engine=ai_engine,
        search_time_limit_ms=search_time_limit,
        hard_move_limit_ms=hard_move_limit,
        random_seed=_require_int(document, "random_seed", maximum=2 ** 32 - 1),
        memory_mode=memory_mode,
        memory_db=_require_text(document, "memory_db", optional=True),
        neural_model_file=_require_text(
            document, "neural_model_file", optional=True
        ),
        neural_model_sha256=_require_text(
            document, "neural_model_sha256", optional=True
        ),
        benchmark_file=_require_text(document, "benchmark_file", optional=True),
        tournament_games=_require_even_positive(document, "tournament_games"),
        tournament_final_games=_require_even_positive(
            document, "tournament_final_games"
        ),
        freeze_commit=_require_text(document, "freeze_commit", optional=True),
        notes=str(document.get("notes", "") or ""),
        profile_path=str(profile_path),
        profile_sha256=file_sha256(profile_path),
    )
    if verify_model:
        _validate_model(document, settings)
    return settings


def _git_output(project_root, *args):
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def working_tree_status(project_root=PROJECT_ROOT):
    return _git_output(Path(project_root), "status", "--porcelain")


def require_clean_worktree(project_root=PROJECT_ROOT):
    status = working_tree_status(project_root)
    if status:
        raise CompetitionConfigError(
            "working tree is dirty; commit changes before a formal evaluation run"
        )
    return _git_output(Path(project_root), "rev-parse", "HEAD")

def reproducibility_metadata(settings=None):
    settings = settings or load_competition_settings()
    status = working_tree_status(PROJECT_ROOT)
    packages = {}
    for name in ("pygame", "torch", "numpy", "PIL"):
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            packages[name] = None
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "code_commit": _git_output(PROJECT_ROOT, "rev-parse", "HEAD"),
        "working_tree_dirty": bool(status),
        "competition_profile": settings.metadata(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }