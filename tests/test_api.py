"""Testes da API FastAPI `/health` e `/predict` (Etapa 3 — tarefa 4)."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from telco_churn.api.main import app
from telco_churn.data.pipeline import build_telco_classifier_pipeline
from telco_churn.data.preprocessing import prepare_telco_features


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_source"] in ("default_synthetic", "joblib_file")


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
    monkeypatch.setenv("TELCO_SKLEARN_PIPELINE_PATH", str(path))
    from telco_churn.api.model_runtime import load_or_fit_serving_pipeline

    model, src = load_or_fit_serving_pipeline()
    assert src == "joblib_file"
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
