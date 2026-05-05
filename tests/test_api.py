"""Testes da API FastAPI: rotas (tarefa 4), logging e latência (tarefa 5)."""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from telco_churn.api.logging_config import JsonLogFormatter
from telco_churn.api.main import app
from telco_churn.data.pipeline import build_telco_classifier_pipeline
from telco_churn.data.preprocessing import prepare_telco_features


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def test_middleware_adds_latency_and_request_id_headers(client: TestClient) -> None:
    r = client.get("/health")
    h = {k.lower(): v for k, v in r.headers.items()}
    assert "x-process-time" in h and "x-request-id" in h
    assert float(h["x-process-time"]) >= 0.0


def test_json_formatter_includes_extra_fields() -> None:
    fmt = JsonLogFormatter()
    rec = logging.LogRecord("telco_churn.test", logging.INFO, __file__, 0, "http_request", (), None)
    rec.request_id = "abc-123"
    rec.latency_ms = 4.2
    line = fmt.format(rec)
    data = json.loads(line)
    assert data["message"] == "http_request"
    assert data["request_id"] == "abc-123"
    assert data["latency_ms"] == 4.2


def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_source"] in (
        "default_synthetic_mlp",
        "mlp_bundle_joblib",
        "sklearn_joblib",
    )


def test_predict_valid_row(client: TestClient) -> None:
    payload = {
        "tenure": 12,
        "MonthlyCharges": 53.85,
        "TotalCharges": "734.35",
        "gender": "Female",
        "Partner": "Yes",
        "PhoneService": "Yes",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "probability_churn" in data and "churn_predicted" in data
    assert 0.0 <= data["probability_churn"] <= 1.0
    assert data["churn_predicted"] in (0, 1)


def test_predict_validation_error_negative_tenure(client: TestClient) -> None:
    r = client.post(
        "/predict",
        json={"tenure": -1, "MonthlyCharges": 50.0},
    )
    assert r.status_code == 422


def test_predict_validation_error_missing_monthly(client: TestClient) -> None:
    r = client.post("/predict", json={"tenure": 5})
    assert r.status_code == 422


def test_load_mlp_bundle_from_joblib_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from telco_churn.api.mlp_predictor import save_mlp_predictor
    from telco_churn.api.model_runtime import (
        fit_default_synthetic_mlp,
        load_or_fit_serving_pipeline,
    )

    pred = fit_default_synthetic_mlp(seed=3)
    bundle_path = tmp_path / "telco_mlp.joblib"
    save_mlp_predictor(bundle_path, pred)
    monkeypatch.setenv("TELCO_MLP_BUNDLE_PATH", str(bundle_path))
    monkeypatch.delenv("TELCO_SKLEARN_PIPELINE_PATH", raising=False)
    model, src = load_or_fit_serving_pipeline()
    assert src == "mlp_bundle_joblib"
    row = pd.DataFrame(
        [
            {
                "tenure": 12.0,
                "MonthlyCharges": 55.0,
                "TotalCharges": 600.0,
                "gender": "Male",
                "Partner": "Yes",
                "PhoneService": "Yes",
            }
        ]
    )
    prob = model.predict_proba(row)[0, 1]
    assert 0.0 <= float(prob) <= 1.0


def test_load_pipeline_from_joblib_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sklearn.dummy import DummyClassifier

    rng = np.random.default_rng(1)
    n = 60
    df = pd.DataFrame(
        {
            "tenure": rng.integers(1, 48, size=n),
            "MonthlyCharges": rng.uniform(20, 100, size=n),
            "TotalCharges": rng.uniform(100, 4000, size=n),
            "gender": rng.choice(["Male", "Female"], size=n),
            "Churn": rng.binomial(1, 0.3, size=n),
        }
    )
    X, y = prepare_telco_features(df)
    pipe = build_telco_classifier_pipeline(DummyClassifier(strategy="prior"))
    pipe.fit(X, y)
    path = tmp_path / "pipe.joblib"
    joblib.dump(pipe, path)
    monkeypatch.delenv("TELCO_MLP_BUNDLE_PATH", raising=False)
    monkeypatch.setenv("TELCO_SKLEARN_PIPELINE_PATH", str(path))
    from telco_churn.api.model_runtime import load_or_fit_serving_pipeline

    model, src = load_or_fit_serving_pipeline()
    assert src == "sklearn_joblib"
    row = pd.DataFrame(
        [
            {
                "tenure": 12.0,
                "MonthlyCharges": 55.0,
                "TotalCharges": 600.0,
                "gender": "Male",
            }
        ]
    )
    p = model.predict_proba(row)[0, 1]
    assert 0.0 <= float(p) <= 1.0
