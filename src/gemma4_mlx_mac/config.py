from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .models import DEFAULT_MODEL_ID

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_STATE_FILE = Path("~/.cache/gemma4-mlx-mac/state.json")


def get_default_model() -> str:
    value = os.environ.get("GEMMA4_MLX_MODEL", "").strip()
    return value or DEFAULT_MODEL_ID


def get_default_host() -> str:
    value = os.environ.get("GEMMA4_MLX_HOST", "").strip()
    return value or DEFAULT_HOST


def get_default_port() -> int:
    raw = os.environ.get("GEMMA4_MLX_PORT", "").strip()
    if not raw:
        return DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError:
        return DEFAULT_PORT
    return port if 1 <= port <= 65535 else DEFAULT_PORT


def get_state_file_path() -> Path:
    raw = os.environ.get("GEMMA4_MLX_STATE_FILE", "").strip()
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_STATE_FILE.expanduser()


def get_allowed_write_roots() -> tuple[Path, ...]:
    """Returns safe base directories where API-initiated writes (datasets/downloads) are allowed."""
    raw = os.environ.get("GEMMA4_MLX_ALLOWED_ROOTS", "").strip()
    if raw:
        roots = [
            Path(part).expanduser().resolve()
            for part in raw.split(os.pathsep)
            if part.strip()
        ]
        if roots:
            return tuple(roots)
    candidates = [
        Path.cwd().resolve(),
        Path.home().resolve(),
        Path(tempfile.gettempdir()).resolve(),
        Path("/tmp").resolve(),
    ]
    return tuple(dict.fromkeys(candidates))


def ensure_safe_write_path(target: str | Path, *, label: str = "Path") -> Path:
    """Rejects sensitive system/dotfile targets and ensures writes stay within allowed roots."""
    resolved = Path(target).expanduser().resolve()
    parts = set(resolved.parts)
    blocked_segments = {".ssh", ".gnupg", ".aws", ".kube", ".config"}
    if parts & blocked_segments:
        raise ValueError(f"{label} cannot write inside sensitive directory: {resolved}")

    system_prefixes = (
        "/etc",
        "/private/etc",
        "/System",
        "/Library",
        "/usr",
        "/bin",
        "/sbin",
        "/var/root",
        "/private/var/root",
    )
    resolved_str = str(resolved)
    for prefix in system_prefixes:
        if resolved_str == prefix or resolved_str.startswith(f"{prefix}/"):
            raise ValueError(f"{label} cannot write inside system directory: {resolved}")

    roots = get_allowed_write_roots()
    if not any(resolved == root or root in resolved.parents for root in roots):
        allowed_display = ", ".join(str(r) for r in roots)
        raise ValueError(f"{label} must be within an allowed directory ({allowed_display}).")
    return resolved
