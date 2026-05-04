"""Orquestração do protocolo holdout: baselines sklearn + MLP (Etapa 2 — tarefa 3)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from telco_churn.data.preprocessing import build_feature_preprocessor, infer_column_types
from telco_churn.evaluation.baselines import default_churn_sklearn_models
from telco_churn.evaluation.metrics import compute_binary_metrics
from telco_churn.modeling.mlp import ChurnMLP
from telco_churn.training.train_mlp import TrainConfig, train_churn_mlp

logger = logging.getLogger(__name__)


def _mlp_positive_proba(model: ChurnMLP, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(xt)
        p_list = torch.sigmoid(logits).detach().cpu().flatten().tolist()
    return np.asarray(p_list, dtype=np.float64)


def _sklearn_positive_proba(fitted: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return fitted.predict_proba(X)[:, 1]


def compare_models_holdout(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    random_state: int = 42,
    test_size: float = 0.2,
    mlp_hidden_dims: tuple[int, ...] = (64, 32),
    mlp_train_config: TrainConfig | None = None,
    device: torch.device | str | None = None,
    return_val_artifacts: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """
    Holdout estratificado: baselines (dummy + logística + RF + HGB) vs MLP nas mesmas métricas.

    Se `return_val_artifacts=True`, retorna também `{"y_val": ndarray, "scores": {model: proba}}`
    para análise de custo FP/FN (Etapa 2 — tarefa 4) sem novo split.
    """
    y_array = np.asarray(y).astype(int).ravel()
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_array,
        test_size=test_size,
        stratify=y_array,
        random_state=random_state,
    )
    num_cols, cat_cols = infer_column_types(X_train.columns)
    prep_template = build_feature_preprocessor(num_cols, cat_cols)

    rows: list[dict[str, Any]] = []
    scores_by_model: dict[str, np.ndarray] = {}
    fitted_sklearn: dict[str, Pipeline] = {}

    sklearn_models = default_churn_sklearn_models(random_state)

    for name, estimator in sklearn_models:
        pipe = Pipeline(steps=[("prep", clone(prep_template)), ("model", estimator)])
        pipe.fit(X_train, y_train)
        proba = _sklearn_positive_proba(pipe, X_val)
        scores_by_model[name] = np.asarray(proba, dtype=np.float64)
        metrics = compute_binary_metrics(y_val, proba)
        rows.append({"model": name, **metrics})
        fitted_sklearn[name] = pipe
        logger.debug("evaluated %s %s", name, metrics)

    prep_mlp = clone(prep_template)
    X_train_m = prep_mlp.fit_transform(X_train, y_train)
    X_val_m = prep_mlp.transform(X_val)
    input_dim = int(X_train_m.shape[1])

    mlp = ChurnMLP(
        input_dim=input_dim,
        hidden_dims=mlp_hidden_dims,
        dropout=0.1,
        activation="relu",
    )
    cfg = mlp_train_config or TrainConfig()
    train_out = train_churn_mlp(
        mlp,
        X_train_m.astype(np.float32),
        y_train,
        X_val_m.astype(np.float32),
        y_val,
        config=cfg,
        device=device,
    )
    dev = torch.device(str(train_out["device"]))
    proba_mlp = _mlp_positive_proba(mlp, X_val_m.astype(np.float32), dev)
    scores_by_model["churn_mlp"] = np.asarray(proba_mlp, dtype=np.float64)
    metrics_mlp = compute_binary_metrics(y_val, proba_mlp)
    rows.append({"model": "churn_mlp", **metrics_mlp})
    logger.debug("evaluated churn_mlp %s", metrics_mlp)

    out = pd.DataFrame(rows).set_index("model")
    if return_val_artifacts:
        return out, {
            "y_val": y_val,
            "scores": scores_by_model,
            "mlp_train_out": train_out,
            "mlp_model": mlp,
            "split_meta": {
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "n_features_transformed": input_dim,
                "random_state": int(random_state),
                "test_size": float(test_size),
                "mlp_hidden_dims": tuple(int(x) for x in mlp_hidden_dims),
            },
            "fitted_sklearn": fitted_sklearn,
        }
    return out
