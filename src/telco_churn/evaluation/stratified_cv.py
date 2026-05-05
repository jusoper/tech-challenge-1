"""Avaliação com validação cruzada estratificada k-fold (OOF), alinhada ao notebook Etapa 1."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from telco_churn.data.pipeline import build_telco_feature_transform_pipeline
from telco_churn.evaluation.baselines import default_churn_sklearn_models
from telco_churn.evaluation.metrics import compute_binary_metrics
from telco_churn.modeling.mlp import ChurnMLP
from telco_churn.training.train_mlp import TrainConfig, train_churn_mlp

logger = logging.getLogger(__name__)


def make_stratified_kfold(
    n_splits: int = 5,
    *,
    shuffle: bool = True,
    random_state: int = 42,
) -> StratifiedKFold:
    """Mesma convenção do notebook `01_eda_baselines.ipynb` (seção 4)."""
    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def _mlp_positive_proba(model: ChurnMLP, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(xt)
        p_list = torch.sigmoid(logits).detach().cpu().flatten().tolist()
    return np.asarray(p_list, dtype=np.float64)


def compare_sklearn_baselines_stratified_cv(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    cv: StratifiedKFold | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int | None = 1,
) -> pd.DataFrame:
    """
    Métricas OOF (out-of-fold) via `cross_val_predict` + `compute_binary_metrics`.

    Cada amostra é predita por um modelo treinado sem ela no fold de treino — mesmas
    quatro métricas que o holdout (`roc_auc`, `pr_auc`, `f1`, `accuracy`).
    """
    y_array = np.asarray(y).astype(int).ravel()
    splitter = cv or make_stratified_kfold(n_splits=n_splits, random_state=random_state)
    feature_template = build_telco_feature_transform_pipeline()
    rows: list[dict[str, Any]] = []

    for name, estimator in default_churn_sklearn_models(random_state):
        pipe = Pipeline(
            steps=[("prep", clone(feature_template)), ("model", clone(estimator))],
        )
        oof_proba = cross_val_predict(
            pipe,
            X,
            y_array,
            cv=splitter,
            method="predict_proba",
            n_jobs=n_jobs,
        )[:, 1]
        metrics = compute_binary_metrics(y_array, oof_proba)
        rows.append({"model": name, **metrics})
        logger.debug("cv_oof %s %s", name, metrics)

    return pd.DataFrame(rows).set_index("model")


def compare_models_stratified_cv(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    cv: StratifiedKFold | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    mlp_hidden_dims: tuple[int, ...] = (64, 32),
    mlp_train_config: TrainConfig | None = None,
    device: torch.device | str | None = None,
    include_mlp: bool = True,
    n_jobs_sklearn: int | None = 1,
) -> pd.DataFrame:
    """
    Baselines sklearn (OOF) + opcionalmente MLP com OOF por fold.

    A MLP é treinada em cada fold (pré-processamento fit só no treino do fold),
    probabilidades no conjunto de validação do fold são concatenadas na ordem do índice.
    """
    table_sk = compare_sklearn_baselines_stratified_cv(
        X,
        y,
        cv=cv,
        n_splits=n_splits,
        random_state=random_state,
        n_jobs=n_jobs_sklearn,
    )
    if not include_mlp:
        return table_sk

    y_array = np.asarray(y).astype(int).ravel()
    splitter = cv or make_stratified_kfold(n_splits=n_splits, random_state=random_state)
    feature_template = build_telco_feature_transform_pipeline()
    n_samples = len(X)
    oof_mlp = np.zeros(n_samples, dtype=np.float64)
    base_cfg = mlp_train_config or TrainConfig()

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y_array)):
        prep = clone(feature_template)
        X_tr = X.iloc[train_idx]
        y_tr = y_array[train_idx]
        X_va = X.iloc[val_idx]
        y_va = y_array[val_idx]
        X_tr_m = prep.fit_transform(X_tr, y_tr)
        X_va_m = prep.transform(X_va)
        input_dim = int(X_tr_m.shape[1])
        mlp = ChurnMLP(
            input_dim=input_dim,
            hidden_dims=mlp_hidden_dims,
            dropout=0.1,
            activation="relu",
        )
        cfg = replace(base_cfg, seed=base_cfg.seed + fold_idx)
        train_out = train_churn_mlp(
            mlp,
            X_tr_m.astype(np.float32),
            y_tr,
            X_va_m.astype(np.float32),
            y_va,
            config=cfg,
            device=device,
        )
        dev = torch.device(str(train_out["device"]))
        oof_mlp[val_idx] = _mlp_positive_proba(mlp, X_va_m.astype(np.float32), dev)
        logger.debug("cv_mlp fold=%s epochs=%s", fold_idx, train_out["epochs_run"])

    metrics_mlp = compute_binary_metrics(y_array, oof_mlp)
    row_mlp = pd.DataFrame([{"model": "churn_mlp", **metrics_mlp}]).set_index("model")
    return pd.concat([table_sk, row_mlp])


def stratified_cv_meta(
    *,
    n_splits: int,
    random_state: int,
    shuffle: bool = True,
) -> dict[str, Any]:
    """Metadados para logging (ex.: MLflow) — protocolo explícito."""
    return {
        "protocol": "stratified_kfold_oof",
        "n_splits": int(n_splits),
        "random_state": int(random_state),
        "shuffle": bool(shuffle),
    }
