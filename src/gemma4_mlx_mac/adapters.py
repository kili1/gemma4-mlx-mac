from __future__ import annotations

import re
from pathlib import Path
from threading import RLock

from pydantic import BaseModel

_SAFE_ADAPTER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class AdapterInfo(BaseModel):
    id: str
    path: str
    active: bool = False
    has_weights: bool = False


class AdapterRegistry:
    def __init__(self, root: str | Path = "adapters") -> None:
        self.root = Path(root)
        self._active_adapter: str | None = None
        self._lock = RLock()

    def adapter_path(self, adapter_id: str) -> Path:
        cleaned = adapter_id.strip()
        if not _SAFE_ADAPTER_RE.match(cleaned):
            raise ValueError(
                "Adapter name must start with a letter or digit and use only "
                "letters, digits, '.', '_', or '-'."
            )
        return self.root / cleaned

    def list_adapters(self) -> list[AdapterInfo]:
        with self._lock:
            if not self.root.exists():
                return []

            adapters: list[AdapterInfo] = []
            for path in sorted(self.root.iterdir()):
                if path.is_dir():
                    adapters.append(
                        AdapterInfo(
                            id=path.name,
                            path=str(path),
                            active=path.name == self._active_adapter,
                            has_weights=_has_adapter_weights(path),
                        )
                    )
            return adapters

    def activate(self, adapter_id: str) -> AdapterInfo:
        path = self.adapter_path(adapter_id)
        if not path.is_dir():
            raise FileNotFoundError(f"Adapter directory does not exist: {path.name}")
        with self._lock:
            self._active_adapter = path.name
            return AdapterInfo(
                id=path.name,
                path=str(path),
                active=True,
                has_weights=_has_adapter_weights(path),
            )

    def deactivate(self) -> None:
        with self._lock:
            self._active_adapter = None

    def active_adapter_id(self) -> str | None:
        with self._lock:
            return self._active_adapter

    def active_adapter_path(self) -> str | None:
        with self._lock:
            if self._active_adapter is None:
                return None
            path = self.root / self._active_adapter
            if not path.exists():
                return None
            return str(path)


def _has_adapter_weights(path: Path) -> bool:
    return (path / "adapters.safetensors").exists() or (path / "adapter_config.json").exists()
