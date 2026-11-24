from gemma4_mlx_mac import system
from gemma4_mlx_mac.system import collect_system_info


def test_collect_system_info_returns_core_fields() -> None:
    info = collect_system_info()

    assert info.os_name
    assert info.python_version
    assert info.total_memory_gb >= 0


def test_collect_system_info_requires_mlx_installed_for_mlx_ready(monkeypatch) -> None:
    monkeypatch.setattr(system.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(system.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(system.platform, "mac_ver", lambda: ("15.4", ("", "", ""), "arm64"))
    monkeypatch.setattr(system, "_memory_bytes", lambda: 32 * 1024**3)

    monkeypatch.setattr(system, "is_mlx_available", lambda: False)
    monkeypatch.setattr(system, "is_mlx_vlm_available", lambda: False)
    missing = collect_system_info()
    assert missing.is_apple_silicon is True
    assert missing.mlx_installed is False
    assert missing.mlx_vlm_installed is False
    assert missing.mlx_ready is False
    assert missing.multimodal_ready is False
    assert any("install-mlx" in rec for rec in missing.recommendations)

    monkeypatch.setattr(system, "is_mlx_available", lambda: True)
    monkeypatch.setattr(system, "is_mlx_vlm_available", lambda: True)
    ready = collect_system_info()
    assert ready.mlx_installed is True
    assert ready.mlx_vlm_installed is True
    assert ready.mlx_ready is True
    assert ready.multimodal_ready is True
