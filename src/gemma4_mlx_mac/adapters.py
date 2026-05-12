from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class AdapterInfo(BaseModel):
    id: str
    path: str
    active: bool = False


class AdapterRegistry:
    def __init__(self, root: str | Path = "adapters") -> None:
        self.root = Path(root)
        self._active_adapter: str | None = None

    def list_adapters(self) -> list[AdapterInfo]:
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
                    )
                )
        return adapters

    def activate(self, adapter_id: str) -> AdapterInfo:
        path = self.root / adapter_id
        self._active_adapter = adapter_id
        return AdapterInfo(id=adapter_id, path=str(path), active=True)
