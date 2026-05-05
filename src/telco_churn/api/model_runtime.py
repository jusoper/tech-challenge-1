"""Carrega ou treina modelo da API: MLP (padrão) ou pipeline sklearn (opcional)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from telco_churn.api.mlp_predictor import TelcoMlpPredictor, load_mlp_predictor
from telco_churn.data.pipeline import (
    build_telco_classifier_pipeline,
    build_telco_feature_transform_pipeline,
)
from telco_churn.data.preprocessing import prepare_telco_features
from telco_churn.modeling.mlp import ChurnMLP
from telco_churn.training.train_mlp import TrainConfig, train_churn_mlp

logger = logging.getLogger(__name__)


def fit_default_synthetic_mlp(*, seed: int = 42) -> TelcoMlpPredictor:
    """
    Treina MLP leve em dados sintéticos (CPU) — alinhado ao desafio (rede neural servida).

    Produção: use `TELCO_MLP_BUNDLE_PATH` com artefato salvo (`TelcoMlpPredictor`) ou
    `TELCO_SKLEARN_PIPELINE_PATH` para servir apenas sklearn.
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
    y_arr = np.asarray(y).astype(int).ravel()
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_arr,
        test_size=0.2,
        stratify=y_arr,
        random_state=seed,
    )

    prep = build_telco_feature_transform_pipeline()
    X_train_m = prep.fit_transform(X_train, y_train)
    X_val_m = prep.transform(X_val)
    input_dim = int(X_train_m.shape[1])

    mlp = ChurnMLP(
        input_dim=input_dim,
        hidden_dims=(48, 24),
        dropout=0.1,
        activation="relu",
    )
    cfg = TrainConfig(
        batch_size=64,
        max_epochs=60,
        patience=12,
        learning_rate=1e-3,
        seed=seed,
    )
    train_churn_mlp(
        mlp,
        np.asarray(X_train_m, dtype=np.float32),
        y_train,
        np.asarray(X_val_m, dtype=np.float32),
        y_val,
        config=cfg,
        device="cpu",
    )
    logger.info(
        "model_fit_default_synthetic_mlp",
        extra={"n_rows": n, "n_features_transformed": input_dim},
    )
    return TelcoMlpPredictor(prep, mlp, device="cpu")


def fit_default_synthetic_pipeline(*, seed: int = 42) -> Pipeline:
    """
    Pipeline sklearn só para compatibilidade / testes que exigem regressão logística sintética.

    Preferir `fit_default_synthetic_mlp` como default da API.
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
        "model_fit_default_synthetic_sklearn",
        extra={"n_rows": n, "n_features": int(X.shape[1])},
    )
    return pipe


def load_or_fit_serving_pipeline() -> tuple[Pipeline | TelcoMlpPredictor, str]:
    """
    Retorna `(modelo, model_source)` para `predict_proba`.

    Ordem de precedência:
    1. `TELCO_MLP_BUNDLE_PATH` — joblib com `TelcoMlpPredictor` (MLP treinada + prep).
    2. `TELCO_SKLEARN_PIPELINE_PATH` — joblib com `sklearn.Pipeline` completo.
    3. Fallback: MLP treinada em dados sintéticos (`default_synthetic_mlp`).
    """
    mlp_raw = os.environ.get("TELCO_MLP_BUNDLE_PATH", "").strip()
    if mlp_raw:
        path = Path(mlp_raw).expanduser().resolve()
        if path.is_file():
            model = load_mlp_predictor(path)
            logger.info("model_loaded_mlp_bundle", extra={"path": str(path)})
            return model, "mlp_bundle_joblib"
        logger.warning(
            "model_mlp_bundle_path_missing",
            extra={"TELCO_MLP_BUNDLE_PATH": mlp_raw},
        )

    sk_raw = os.environ.get("TELCO_SKLEARN_PIPELINE_PATH", "").strip()
    if sk_raw:
        path = Path(sk_raw).expanduser().resolve()
        if path.is_file():
            model = joblib.load(path)
            if not isinstance(model, Pipeline):
                raise TypeError("TELCO_SKLEARN_PIPELINE_PATH deve apontar para um sklearn.Pipeline")
            logger.info(
                "model_loaded_sklearn_joblib",
                extra={"path": str(path)},
            )
            return model, "sklearn_joblib"
        logger.warning(
            "model_sklearn_joblib_path_missing",
            extra={"TELCO_SKLEARN_PIPELINE_PATH": sk_raw},
        )

    return fit_default_synthetic_mlp(), "default_synthetic_mlp"
