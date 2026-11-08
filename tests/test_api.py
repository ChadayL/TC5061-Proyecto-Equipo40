
import httpx
import pytest
from fastapi.testclient import TestClient

from mibici.api.main import app

client = TestClient(app)

VALID_INPUT = {
    "station_id": 1,
    "day_of_week": 2,
    "departures_lag_1": 40.0,
    "departures_lag_7": 35.0,
    "departures_mean_7": 38.5,
}


def mock_response(request, status_code=200, payload=None):
    if payload is None:
        payload = {"predictions": [46.3]}

    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request("POST", str(request.url)),
    )


def test_home_redirects_to_docs():
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_predict_valid_input(monkeypatch):
    def fake_post(url, json, timeout):
        assert json == {"dataframe_records": [VALID_INPUT]}
        assert timeout == 10.0
        return mock_response(httpx.Request("POST", url))

    monkeypatch.setattr("mibici.api.main.httpx.post", fake_post)

    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 200
    assert response.json() == {"predicted_departures": 46.3}


def test_predict_missing_field():
    payload = VALID_INPUT.copy()
    del payload["station_id"]

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_station():
    payload = {**VALID_INPUT, "station_id": 0}

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_day():
    payload = {**VALID_INPUT, "day_of_week": 7}

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_departures():
    payload = {**VALID_INPUT, "departures_lag_1": -5.0}

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_extra_field():
    payload = {**VALID_INPUT, "unknown_feature": 123}

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_timeout(monkeypatch):
    def fake_post(url, json, timeout):
        raise httpx.ReadTimeout("Timeout")

    monkeypatch.setattr("mibici.api.main.httpx.post", fake_post)

    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 504


def test_predict_connection_error(monkeypatch):
    def fake_post(url, json, timeout):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr("mibici.api.main.httpx.post", fake_post)

    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 502


def test_predict_invalid_inference_response(monkeypatch):
    def fake_post(url, json, timeout):
        return mock_response(
            httpx.Request("POST", url),
            payload={"predictions": []},
        )

    monkeypatch.setattr("mibici.api.main.httpx.post", fake_post)

    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 502

def test_predict_reproducibility(monkeypatch):
    """La misma entrada debe producir la misma predicción."""

    calls = []

    def fake_post(url, json, timeout):
        calls.append(json)
        return mock_response(
            httpx.Request("POST", url),
            payload={"predictions": [46.3]},
        )

    monkeypatch.setattr("mibici.api.main.httpx.post", fake_post)

    response_1 = client.post("/predict", json=VALID_INPUT)
    response_2 = client.post("/predict", json=VALID_INPUT)

    assert response_1.status_code == 200
    assert response_2.status_code == 200
    assert response_1.json() == response_2.json()
    assert len(calls) == 2
    assert calls[0] == calls[1]
