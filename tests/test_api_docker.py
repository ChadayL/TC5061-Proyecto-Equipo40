"""Pruebas de integración HTTP contra la API ejecutada en Docker."""

import os

import httpx
import pytest

API_URL = os.getenv("MIBICI_API_URL", "http://127.0.0.1:8000")


@pytest.fixture(scope="module")
def client():
    """Verifica que la API esté disponible antes de ejecutar las pruebas."""
    try:
        response = httpx.get(f"{API_URL}/openapi.json", timeout=5)
        response.raise_for_status()
    except httpx.HTTPError:
        pytest.skip("API Docker no disponible")
    with httpx.Client(base_url=API_URL, timeout=5) as http_client:
        yield http_client


def test_docker_docs(client):
    response = client.get("/docs")
    assert response.status_code == 200


def test_docker_openapi(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/predict" in schema["paths"]
    assert "post" in schema["paths"]["/predict"]


def test_docker_invalid_values(client):
    payload = {
        "station_id": 0,
        "day_of_week": 8,
        "departures_lag_1": -5,
        "departures_lag_7": 21,
        "departures_mean_7": 55,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    invalid_fields = {error["loc"][-1] for error in errors}
    assert invalid_fields == {
        "station_id",
        "day_of_week",
        "departures_lag_1",
    }


def test_docker_missing_required_field(client):
    payload = {
        "station_id": 2,
        "day_of_week": 1,
        "departures_lag_1": 76,
        "departures_lag_7": 21,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert any(
        error["loc"][-1] == "departures_mean_7"
        for error in response.json()["detail"]
    )
