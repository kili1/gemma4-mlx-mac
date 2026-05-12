from __future__ import annotations

import os
import platform
import subprocess
import sys

from pydantic import BaseModel, Field


class SystemInfo(BaseModel):
    os_name: str
    os_version: str
    machine: str
    python_version: str
    total_memory_gb: float
    is_macos: bool
    is_apple_silicon: bool
    mlx_ready: bool
    recommendations: list[str] = Field(default_factory=list)


def _memory_bytes() -> int:
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
            )
            return int(result.stdout.strip())
        except (OSError, subprocess.CalledProcessError, ValueError):
            return 0

    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size)
        except (OSError, ValueError):
            return 0

    return 0


def collect_system_info() -> SystemInfo:
    os_name = platform.system()
    machine = platform.machine()
    total_memory_gb = round(_memory_bytes() / (1024**3), 1)
    is_macos = os_name == "Darwin"
    is_apple_silicon = is_macos and machine == "arm64"
    recommendations: list[str] = []

    if not is_macos:
        recommendations.append("MLX acceleration requires macOS on Apple Silicon.")
    elif not is_apple_silicon:
        recommendations.append("MLX acceleration is designed for Apple Silicon Macs.")

    if total_memory_gb and total_memory_gb < 16:
        recommendations.append("Use the E2B 4-bit profile and short contexts on low-memory Macs.")
    elif total_memory_gb and total_memory_gb < 32:
        recommendations.append("Start with Gemma 4 E2B 4-bit before trying larger models.")

    if is_macos:
        major_version = _macos_major_version(platform.mac_ver()[0])
        if major_version and major_version < 15:
            recommendations.append("macOS 15 or newer is recommended for large MLX models.")

    return SystemInfo(
        os_name=os_name,
        os_version=platform.platform(),
        machine=machine,
        python_version=sys.version.split()[0],
        total_memory_gb=total_memory_gb,
        is_macos=is_macos,
        is_apple_silicon=is_apple_silicon,
        mlx_ready=is_apple_silicon,
        recommendations=recommendations,
    )


def _macos_major_version(version: str) -> int | None:
    if not version:
        return None
    try:
        return int(version.split(".", maxsplit=1)[0])
    except ValueError:
        return None
