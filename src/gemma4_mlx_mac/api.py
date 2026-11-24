from __future__ import annotations

import json
import time
from collections.abc import Iterable

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .adapters import AdapterRegistry
from .config import get_default_model
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
adapters = AdapterRegistry()
model_downloads = ModelDownloadJobStore()
mlx_installer = MlxInstallerJobStore()


def _resolve_model_path(model_id: str) -> str | None:
    return model_downloads.downloaded_paths().get(model_id) or get_cached_model_path(model_id)


chat_service = ChatService(
    model_path_resolver=_resolve_model_path,
    active_adapter_resolver=adapters.active_adapter_path,
)
tune_jobs = TuneJobStore(
    adapter_registry=adapters,
    model_path_resolver=_resolve_model_path,
)


def shutdown_services() -> None:
    tune_jobs.shutdown()
    model_downloads.shutdown()
    mlx_installer.shutdown()
    chat_service.unload_models()


@router.get("/api/system")
def get_system() -> dict:
    payload = collect_system_info().model_dump()
    payload["default_model"] = get_default_model()
    return payload


@router.get("/api/models")
def get_models() -> dict:
    downloaded_paths = model_downloads.downloaded_paths()
    configured_default = get_default_model()
    profiles = list_model_profiles()
    has_configured_match = any(p.id == configured_default for p in profiles)
    return {
        "models": [
            _with_download_state(
                profile,
                downloaded_paths,
                is_default=(profile.id == configured_default)
                if has_configured_match
                else profile.default,
            ).model_dump()
            for profile in profiles
        ]
    }


@router.get("/v1/models")
def list_openai_models() -> dict:
    """OpenAI-compatible model catalog endpoint for external clients."""
    downloaded_paths = model_downloads.downloaded_paths()
    created = int(time.time())
    data: list[dict] = []
    seen: set[str] = set()
    for profile in list_model_profiles():
        local_path = downloaded_paths.get(profile.id) or get_cached_model_path(profile.id)
        seen.add(profile.id)
        data.append(
            {
                "id": profile.id,
                "object": "model",
                "created": created,
                "owned_by": profile.id.split("/", maxsplit=1)[0]
                if "/" in profile.id
                else "local",
                "downloaded": local_path is not None,
            }
        )
    for model_id in downloaded_paths:
        if model_id not in seen:
            data.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": "local",
                    "downloaded": True,
                }
            )
    return {"object": "list", "data": data}


def _with_download_state(
    profile: ModelProfile,
    downloaded_paths: dict[str, str],
    *,
    is_default: bool | None = None,
) -> ModelProfile:
    local_path = downloaded_paths.get(profile.id) or get_cached_model_path(profile.id)
    update: dict[str, object] = {
        "downloaded": local_path is not None,
        "local_path": local_path,
    }
    if is_default is not None:
        update["default"] = is_default
    return profile.model_copy(update=update)


@router.post("/api/models/download")
def download_model(request: ModelDownloadRequest) -> dict:
    try:
        job = model_downloads.start(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return job.model_dump(mode="json")


@router.get("/api/models/download/{job_id}")
def get_model_download(job_id: str) -> dict:
    job = model_downloads.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Download job not found.")
    return job.model_dump(mode="json")


@router.get("/api/models/downloads")
def list_model_downloads() -> dict:
    return {"downloads": [job.model_dump(mode="json") for job in model_downloads.list()]}


@router.post("/api/models/unload")
def unload_models() -> dict:
    unloaded = chat_service.unload_models()
    return {
        "unloaded": unloaded,
        "model_memory": chat_service.memory_status().model_dump(mode="json"),
    }


@router.get("/api/inference/status")
def get_inference_status() -> dict:
    status = mlx_installer.status().model_dump(mode="json")
    status["model_memory"] = chat_service.memory_status().model_dump(mode="json")
    status["active_adapter"] = adapters.active_adapter_id()
    return status


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
        prepare = getattr(chat_service, "prepare_request", None)
        if callable(prepare):
            try:
                prepare(request)
            except InferenceNotReady as exc:
                return _error_response(503, "not_ready", str(exc))
            except InferenceError as exc:
                return _error_response(500, "inference_error", str(exc))
        return StreamingResponse(
            _stream_chat_completion(request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        return JSONResponse(chat_service.create_completion(request))
    except InferenceNotReady as exc:
        return _error_response(503, "not_ready", str(exc))
    except InferenceError as exc:
        return _error_response(500, "inference_error", str(exc))


def _error_response(status_code: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "type": error_type,
                "message": message,
            }
        },
    )


def _stream_chat_completion(request: ChatCompletionRequest) -> Iterable[str]:
    completion_id = f"chatcmpl-stream-{int(time.time() * 1000)}"
    created = int(time.time())
    last_chunk: GeneratedToken | None = None
    first_chunk = True
    include_usage = bool(
        isinstance(request.stream_options, dict) and request.stream_options.get("include_usage")
    )
    try:
        for chunk in chat_service.stream_tokens(request):
            last_chunk = chunk
            delta: dict[str, str] = {"content": chunk.text}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            yield _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": None,
                        }
                    ],
                    "metrics": _generation_metrics(chunk),
                }
            )
        final_payload: dict[str, object] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": (last_chunk.finish_reason if last_chunk else None) or "stop",
                }
            ],
            "metrics": _generation_metrics(last_chunk) if last_chunk else None,
        }
        if include_usage and last_chunk is not None:
            final_payload["usage"] = {
                "prompt_tokens": last_chunk.prompt_tokens,
                "completion_tokens": last_chunk.completion_tokens,
                "total_tokens": last_chunk.prompt_tokens + last_chunk.completion_tokens,
            }
        yield _sse(final_payload)
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


@router.get("/api/tunes")
def list_tunes() -> dict:
    return {"tunes": [job.model_dump(mode="json") for job in tune_jobs.list()]}


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


@router.post("/api/tunes/{job_id}/cancel")
def cancel_tune(job_id: str) -> dict:
    job = tune_jobs.cancel(job_id)
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
    try:
        info = adapters.activate(adapter_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    chat_service.unload_models()
    return info.model_dump()


@router.post("/api/adapters/deactivate")
def deactivate_adapter() -> dict:
    adapters.deactivate()
    chat_service.unload_models()
    return {"active": None}


@router.get("/api/adapters")
def list_adapters() -> dict:
    return {"adapters": [adapter.model_dump() for adapter in adapters.list_adapters()]}
