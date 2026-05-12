from fastapi.testclient import TestClient

from gemma4_mlx_mac.app import create_app
from gemma4_mlx_mac.models import DEFAULT_MODEL_ID


def test_health_system_and_models_routes() -> None:
    client = TestClient(create_app())

    system_response = client.get("/api/system")
    models_response = client.get("/api/models")

    assert system_response.status_code == 200
    assert models_response.status_code == 200
    assert models_response.json()["models"][0]["id"] == DEFAULT_MODEL_ID


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
