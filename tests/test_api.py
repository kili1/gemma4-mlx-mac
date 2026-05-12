from time import sleep

from fastapi.testclient import TestClient

from gemma4_mlx_mac import api
from gemma4_mlx_mac.app import create_app
from gemma4_mlx_mac.downloads import ModelDownloader, ModelDownloadJobStore
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
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    def fake_snapshot_download(**kwargs: object) -> str:
        assert kwargs["repo_id"] == DEFAULT_MODEL_ID
        return str(model_dir)

    monkeypatch.setattr(
        api,
        "model_downloads",
        ModelDownloadJobStore(ModelDownloader(fake_snapshot_download), max_workers=1),
    )
    client = TestClient(create_app())

    response = client.post("/api/models/download", json={"model": DEFAULT_MODEL_ID})

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


def test_chat_route_shape_returns_not_implemented() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": DEFAULT_MODEL_ID,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 501
    assert response.json()["error"]["type"] == "not_implemented"


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
