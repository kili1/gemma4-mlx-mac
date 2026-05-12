from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from .adapters import AdapterRegistry
from .downloads import ModelDownloadJobStore, ModelDownloadRequest, get_cached_model_path
from .inference import ChatCompletionRequest, ChatService, InferenceNotReady
from .models import ModelProfile, list_model_profiles
from .system import collect_system_info
from .tuning import TuneJobStore, TuneRequest

router = APIRouter()
chat_service = ChatService()
tune_jobs = TuneJobStore()
adapters = AdapterRegistry()
model_downloads = ModelDownloadJobStore()


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


@router.post("/v1/chat/completions")
def create_chat_completion(request: ChatCompletionRequest) -> JSONResponse:
    try:
        return JSONResponse(chat_service.create_completion(request))
    except InferenceNotReady as exc:
        return JSONResponse(
            status_code=501,
            content={
                "error": {
                    "type": "not_implemented",
                    "message": str(exc),
                }
            },
        )


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


@router.post("/api/adapters/{adapter_id}/activate")
def activate_adapter(adapter_id: str) -> dict:
    return adapters.activate(adapter_id).model_dump()
