from __future__ import annotations

from pathlib import Path

import pytest

from gemma4_mlx_mac.adapters import AdapterRegistry


def test_adapter_registry_lists_and_activates_existing_adapter(tmp_path: Path) -> None:
    root = tmp_path / "adapters"
    adapter_dir = root / "demo-lora"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapters.safetensors").write_bytes(b"weights")

    registry = AdapterRegistry(root)
    listed = registry.list_adapters()
    assert len(listed) == 1
    assert listed[0].id == "demo-lora"
    assert listed[0].active is False
    assert listed[0].has_weights is True

    activated = registry.activate("demo-lora")
    assert activated.active is True
    assert activated.has_weights is True
    assert registry.active_adapter_id() == "demo-lora"
    assert registry.active_adapter_path() == str(adapter_dir)

    registry.deactivate()
    assert registry.active_adapter_id() is None
    assert registry.active_adapter_path() is None


def test_adapter_registry_rejects_missing_and_unsafe_names(tmp_path: Path) -> None:
    registry = AdapterRegistry(tmp_path / "adapters")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        registry.activate("nonexistent")

    with pytest.raises(ValueError, match="Adapter name"):
        registry.activate("../escape")
