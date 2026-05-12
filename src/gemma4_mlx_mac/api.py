from __future__ import annotations

import json
import time
from collections.abc import Iterable

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .adapters import AdapterRegistry
from .datasets import SyntheticDatasetRequest, create_synthetic_dataset
from .downloads import ModelDownloadJobStore, ModelDownloadRequest, get_cached_model_path
from .inference import (
    ChatCompletionRequest,
    ChatService,
    GeneratedToken,
    InferenceError,
    InferenceNotReady,
)
from .mlx_setup import MlxInstallerJobStore
from .models import ModelProfile, list_model_profiles
from .system import collect_system_info
from .tuning import TuneJobStore, TuneRequest

router = APIRouter()
chat_service = ChatService()
tune_jobs = TuneJobStore()
adapters = AdapterRegistry()
model_downloads = ModelDownloadJobStore()
mlx_installer = MlxInstallerJobStore()


@router.get("/api/system")
def get_system() -> dict:
    return collect_system_info().model_dump()


@router.get("/api/models")
def get_models() -> dict:
    downloaded_paths = model_downloads.downloaded_paths()
    return {
        "models": [
            _with_download_state(profile, downloaded_paths).model_dump()
            for profile in list_model_profiles()
        ]
    }


def _with_download_state(
    profile: ModelProfile,
    downloaded_paths: dict[str, str],
) -> ModelProfile:
    local_path = downloaded_paths.get(profile.id) or get_cached_model_path(profile.id)
    return profile.model_copy(
        update={
            "downloaded": local_path is not None,
            "local_path": local_path,
        }
    )


@router.post("/api/models/download")
def download_model(request: ModelDownloadRequest) -> dict:
    return model_downloads.start(request).model_dump(mode="json")


@router.get("/api/models/download/{job_id}")
def get_model_download(job_id: str) -> dict:
    job = model_downloads.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Download job not found.")
    return job.model_dump(mode="json")


@router.get("/api/models/downloads")
def list_model_downloads() -> dict:
    return {"downloads": [job.model_dump(mode="json") for job in model_downloads.list()]}


@router.get("/api/inference/status")
def get_inference_status() -> dict:
    return mlx_installer.status().model_dump(mode="json")


@router.post("/api/inference/install")
def install_inference_dependencies() -> dict:
    return mlx_installer.start().model_dump(mode="json")


@router.get("/api/inference/install/{job_id}")
def get_inference_install(job_id: str) -> dict:
    job = mlx_installer.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="MLX install job not found.")
    return job.model_dump(mode="json")


@router.post("/v1/chat/completions")
def create_chat_completion(request: ChatCompletionRequest):
    if request.stream:
        return StreamingResponse(
            _stream_chat_completion(request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        return JSONResponse(chat_service.create_completion(request))
    except InferenceNotReady as exc:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": "not_ready",
                    "message": str(exc),
                }
            },
        )
    except InferenceError as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "inference_error",
                    "message": str(exc),
                }
            },
        )


def _stream_chat_completion(request: ChatCompletionRequest) -> Iterable[str]:
    completion_id = f"chatcmpl-stream-{int(time.time() * 1000)}"
    created = int(time.time())
    last_chunk: GeneratedToken | None = None
    try:
        for chunk in chat_service.stream_tokens(request):
            last_chunk = chunk
            yield _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk.text},
                            "finish_reason": None,
                        }
                    ],
                    "metrics": _generation_metrics(chunk),
                }
            )
        yield _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": (last_chunk.finish_reason if last_chunk else None)
                        or "stop",
                    }
                ],
                "metrics": _generation_metrics(last_chunk) if last_chunk else None,
            }
        )
    except InferenceNotReady as exc:
        yield _sse({"error": {"type": "not_ready", "message": str(exc)}})
    except InferenceError as exc:
        yield _sse({"error": {"type": "inference_error", "message": str(exc)}})
    yield "data: [DONE]\n\n"


def _generation_metrics(chunk: GeneratedToken) -> dict:
    return {
        "prompt_tokens": chunk.prompt_tokens,
        "completion_tokens": chunk.completion_tokens,
        "elapsed_seconds": round(chunk.elapsed_seconds, 3),
        "tokens_per_second": round(chunk.tokens_per_second, 2),
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/api/tunes")
def create_tune(request: TuneRequest) -> dict:
    try:
        job = tune_jobs.create(request)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return job.model_dump(mode="json")


@router.get("/api/tunes/{job_id}")
def get_tune(job_id: str) -> dict:
    job = tune_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Tune job not found.")
    return job.model_dump(mode="json")


@router.post("/api/datasets/synthetic")
def create_synthetic_data(request: SyntheticDatasetRequest) -> dict:
    try:
        result = create_synthetic_dataset(request)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump(mode="json")


@router.post("/api/adapters/{adapter_id}/activate")
def activate_adapter(adapter_id: str) -> dict:
    return adapters.activate(adapter_id).model_dump()


@router.get("/api/adapters")
def list_adapters() -> dict:
    return {"adapters": [adapter.model_dump() for adapter in adapters.list_adapters()]}
