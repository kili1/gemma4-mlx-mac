"""Local Gemma 4 on Apple Silicon with MLX."""

from importlib.metadata import PackageNotFoundError, version

from .models import DEFAULT_MODEL_ID

__all__ = ["DEFAULT_MODEL_ID", "__version__"]

try:
    __version__ = version("gemma4-mlx-mac")
except PackageNotFoundError:  # pragma: no cover - fallback when running from raw source tree.
    __version__ = "0.1.0"
