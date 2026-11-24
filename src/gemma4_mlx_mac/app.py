from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import chat_service, router, shutdown_services
from .config import get_default_model
from .inference import InferenceError, InferenceNotReady

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "testserver"})
_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _extract_host(raw_host: str) -> str:
    host = raw_host.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        return host[1:end] if end != -1 else host
    return host.split(":", maxsplit=1)[0]


def _is_local_origin(origin: str) -> bool:
    parsed = urlparse(origin.strip())
    if parsed.scheme not in {"http", "https"}:
        return False
    return (parsed.hostname or "").lower() in _LOCAL_HOSTS


def create_app(*, preload_model: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        target_preload = preload_model
        if target_preload is None and os.environ.get("GEMMA4_MLX_PRELOAD") == "1":
            target_preload = get_default_model()
        if target_preload:
            try:
                chat_service.preload(target_preload)
            except (InferenceNotReady, InferenceError) as exc:
                logger.warning("Skipping preload for %s: %s", target_preload, exc)
        try:
            yield
        finally:
            shutdown_services()

    app = FastAPI(
        title="gemma4-mlx-mac",
        version=__version__,
        summary="Local Gemma 4 on Apple Silicon with MLX.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def local_only_guard(request: Request, call_next):
        if os.environ.get("GEMMA4_MLX_ALLOW_REMOTE") != "1":
            host_header = request.headers.get("host")
            if host_header and _extract_host(host_header) not in _LOCAL_HOSTS:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Rejected non-local Host header."},
                )
            if request.method in _MUTATING_METHODS:
                origin = request.headers.get("origin")
                if origin and not _is_local_origin(origin):
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "Rejected cross-origin request to local API."},
                    )
        return await call_next(request)

    app.include_router(router)
    assets = files("gemma4_mlx_mac") / "web" / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _load_index_html()

    return app


def _load_index_html() -> str:
    try:
        return (files("gemma4_mlx_mac") / "web" / "index.html").read_text(encoding="utf-8")
    except FileNotFoundError:
        return "<!doctype html><title>gemma4-mlx-mac</title><h1>gemma4-mlx-mac</h1>"


app = create_app()
