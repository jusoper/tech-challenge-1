"""Carrega ou treina pipeline sklearn servido pela API (Etapa 3 — tarefa 4)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from telco_churn.data.pipeline import build_telco_classifier_pipeline
from telco_churn.data.preprocessing import prepare_telco_features

logger = logging.getLogger(__name__)


def fit_default_synthetic_pipeline(*, seed: int = 42) -> Pipeline:
    """
    Treina um pipeline leve em dados sintéticos (CPU) para desenvolvimento e testes.

    Produção: defina `TELCO_SKLEARN_PIPELINE_PATH` apontando para um `.joblib` salvo
    (ex.: pipeline sklearn exportado após treino).
    """
    rng = np.random.default_rng(seed)
    n = 400
    df = pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, size=n),
            "MonthlyCharges": rng.normal(65.0, 25.0, size=n).clip(18.0, 120.0),
            "TotalCharges": rng.normal(2500, 800, size=n).clip(0, None),
            "gender": rng.choice(["Male", "Female"], size=n),
            "Partner": rng.choice(["Yes", "No"], size=n),
            "PhoneService": rng.choice(["Yes", "No"], size=n),
            "Churn": rng.binomial(1, 0.27, size=n),
        }
    )
    X, y = prepare_telco_features(df)
    pipe = build_telco_classifier_pipeline(
        LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced"),
    )
    pipe.fit(X, y)
    logger.info(
        "model_fit_default_synthetic",
        extra={"n_rows": n, "n_features": int(X.shape[1])},
    )
    return pipe


def load_or_fit_serving_pipeline() -> tuple[Pipeline, str]:
    """
    Retorna `(pipeline, model_source)`.

    Ordem: variável de ambiente `TELCO_SKLEARN_PIPELINE_PATH` (joblib) ou baseline sintético.
    """
    raw = os.environ.get("TELCO_SKLEARN_PIPELINE_PATH", "").strip()
    if raw:
        path = Path(raw).expanduser().resolve()
        if path.is_file():
            model = joblib.load(path)
            if not isinstance(model, Pipeline):
                raise TypeError("TELCO_SKLEARN_PIPELINE_PATH deve apontar para um sklearn.Pipeline")
            logger.info(
                "model_loaded_joblib",
                extra={"path": str(path)},
            )
            return model, "joblib_file"
        logger.warning(
            "model_joblib_path_missing",
            extra={"TELCO_SKLEARN_PIPELINE_PATH": raw},
        )
    return fit_default_synthetic_pipeline(), "default_synthetic"
