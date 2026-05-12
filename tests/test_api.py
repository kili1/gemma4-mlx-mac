from time import sleep

from fastapi.testclient import TestClient

from gemma4_mlx_mac import api
from gemma4_mlx_mac.app import create_app
from gemma4_mlx_mac.downloads import ModelDownloader, ModelDownloadJobStore
from gemma4_mlx_mac.mlx_setup import MlxInstallJob, MlxStatus
from gemma4_mlx_mac.models import DEFAULT_MODEL_ID


def test_health_system_and_models_routes() -> None:
    client = TestClient(create_app())

    system_response = client.get("/api/system")
    models_response = client.get("/api/models")

    assert system_response.status_code == 200
    assert models_response.status_code == 200
    assert models_response.json()["models"][0]["id"] == DEFAULT_MODEL_ID


def test_model_download_route_uses_huggingface_snapshot(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "snapshot"
    target_dir = tmp_path / "downloads"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    def fake_snapshot_download(**kwargs: object) -> str:
        assert kwargs["repo_id"] == DEFAULT_MODEL_ID
        assert kwargs["local_dir"] == str(target_dir)
        return str(model_dir)

    monkeypatch.setattr(
        api,
        "model_downloads",
        ModelDownloadJobStore(ModelDownloader(fake_snapshot_download), max_workers=1),
    )
    client = TestClient(create_app())

    response = client.post(
        "/api/models/download",
        json={"model": DEFAULT_MODEL_ID, "local_dir": str(target_dir)},
    )

    assert response.status_code == 200
    job_id = response.json()["id"]

    job_response = client.get(f"/api/models/download/{job_id}")
    assert job_response.status_code == 200
    assert job_response.json()["status"] in {"queued", "running", "succeeded"}

    for _ in range(20):
        job_response = client.get(f"/api/models/download/{job_id}")
        if job_response.json()["status"] == "succeeded":
            break
        sleep(0.01)

    assert job_response.json()["path"] == str(model_dir)

    models_response = client.get("/api/models")
    default_model = models_response.json()["models"][0]
    assert default_model["downloaded"] is True
    assert default_model["local_path"] == str(model_dir)


def test_frontend_shell_is_served() -> None:
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "gemma4-mlx-mac" in response.text


def test_inference_status_and_install_routes(monkeypatch) -> None:
    job = MlxInstallJob(
        id="job-1",
        status="running",
        command=["uv", "pip", "install", "gemma4-mlx-mac[mlx]"],
        message="Installing MLX inference dependencies...",
        created_at="2026-05-12T00:00:00Z",
    )

    class FakeInstaller:
        def status(self) -> MlxStatus:
            return MlxStatus(
                available=False,
                installing=True,
                install_job_id=job.id,
                command=job.command,
                message="Installing MLX inference dependencies...",
                job=job,
            )

        def start(self) -> MlxInstallJob:
            return job

        def get(self, job_id: str) -> MlxInstallJob | None:
            return job if job_id == job.id else None

    monkeypatch.setattr(api, "mlx_installer", FakeInstaller())
    client = TestClient(create_app())

    status_response = client.get("/api/inference/status")
    install_response = client.post("/api/inference/install")
    job_response = client.get(f"/api/inference/install/{job.id}")

    assert status_response.status_code == 200
    assert status_response.json()["available"] is False
    assert install_response.status_code == 200
    assert install_response.json()["id"] == job.id
    assert job_response.status_code == 200
    assert job_response.json()["status"] == "running"


def test_chat_route_shape_returns_completion(monkeypatch) -> None:
    class FakeChatService:
        def create_completion(self, request) -> dict:
            assert request.model == DEFAULT_MODEL_ID
            return {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello back"},
                        "finish_reason": "stop",
                    }
                ],
            }

    monkeypatch.setattr(api, "chat_service", FakeChatService())
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": DEFAULT_MODEL_ID,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hello back"


def test_tune_job_route_validates_dataset() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/tunes",
        json={
            "model": DEFAULT_MODEL_ID,
            "data_path": "examples/data",
            "adapter_name": "demo",
            "iters": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "queued"


def test_adapters_route_lists_registry() -> None:
    client = TestClient(create_app())

    response = client.get("/api/adapters")

    assert response.status_code == 200
    assert "adapters" in response.json()
