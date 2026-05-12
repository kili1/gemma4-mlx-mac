from __future__ import annotations

from importlib.resources import files

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .api import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="gemma4-mlx-mac",
        version="0.1.0",
        summary="Local Gemma 4 on Apple Silicon with MLX.",
    )
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
