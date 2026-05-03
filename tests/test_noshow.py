import pytest
from fastapi.testclient import TestClient
from noshow_iq.api import app

client = TestClient(app)


SAMPLE_PAYLOAD = {
    "age": 30,
    "scholarship": 0,
    "hypertension": 0,
    "diabetes": 0,
    "alcoholism": 0,
    "handicap": 0,
    "sms_received": 1,
    "days_in_advance": 5,
    "appt_day_of_week": 2,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_200():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200


def test_predict_has_risk_level():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    data = response.json()
    assert "risk_level" in data
    assert data["risk_level"] in ["high", "low"]


def test_predict_has_probability():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    data = response.json()
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_has_recommendation():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    data = response.json()
    assert "recommendation" in data


def test_history_returns_list():
    response = client.get("/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_stats_returns_correct_keys():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "high_risk_count" in data
    assert "low_risk_count" in data
    assert "average_probability" in data
