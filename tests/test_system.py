from gemma4_mlx_mac.system import collect_system_info


def test_collect_system_info_returns_core_fields() -> None:
    info = collect_system_info()

    assert info.os_name
    assert info.python_version
    assert info.total_memory_gb >= 0
