"""
Basic tests for the churn prediction API.

Run:
    pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture()
def client():
    # Using the context manager form ensures the lifespan startup handler
    # (which loads the model artifact) actually runs before tests execute.
    with TestClient(app) as c:
        yield c

HIGH_RISK_CUSTOMER = {
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure_months": 3,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 89.5,
    "total_charges": 268.5,
}

LOW_RISK_CUSTOMER = {
    **HIGH_RISK_CUSTOMER,
    "tenure_months": 60,
    "contract": "Two year",
    "internet_service": "DSL",
    "online_security": "Yes",
    "tech_support": "Yes",
    "monthly_charges": 55.0,
    "total_charges": 3300.0,
}


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_model_info(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert "threshold" in body
    assert "trained_at" in body


def test_predict_high_risk(client):
    resp = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in ("Yes", "No")


def test_predict_low_risk_scores_lower_than_high_risk(client):
    """Sanity check that the model has learned a sensible direction:
    a loyal, long-tenure, well-supported customer should score meaningfully
    lower churn risk than a new, unsupported, month-to-month customer.
    """
    high = client.post("/predict", json=HIGH_RISK_CUSTOMER).json()
    low = client.post("/predict", json=LOW_RISK_CUSTOMER).json()
    assert low["churn_probability"] < high["churn_probability"]


def test_predict_batch(client):
    resp = client.post("/predict/batch", json=[HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER])
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2


def test_predict_rejects_invalid_input(client):
    bad_customer = {**HIGH_RISK_CUSTOMER, "contract": "Lifetime"}  # not a valid category
    resp = client.post("/predict", json=bad_customer)
    assert resp.status_code == 422
