"""Registro no MLflow da comparação holdout — MLP e baselines/ensembles (Etapa 2 — tarefa 5)."""

from __future__ import annotations

import importlib
import logging
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from telco_churn.evaluation.holdout import compare_models_holdout
from telco_churn.training.train_mlp import TrainConfig

logger = logging.getLogger(__name__)


def _normalize_tracking_uri(tracking_uri: str | Path) -> str:
    s = str(tracking_uri)
    if s.startswith("file:") or "://" in s:
        return s
    return f"file:{Path(tracking_uri).resolve()}"


def _mlflow_safe_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if v is None:
        return "null"
    if isinstance(v, tuple):
        return ",".join(str(x) for x in v)[:250]
    return str(v)[:250]


def _flatten_params(prefix: str, data: dict[str, Any]) -> dict[str, str]:
    pre = f"{prefix}." if prefix else ""
    return {f"{pre}{k}"[:250]: _mlflow_safe_value(v) for k, v in data.items()}


def _simple_estimator_params(estimator: Any, *, max_keys: int = 28) -> dict[str, Any]:
    """Subconjunto de `get_params(deep=False)` com tipos simples (evita explosão de chaves)."""
    if not hasattr(estimator, "get_params"):
        return {}
    out: dict[str, Any] = {}
    for k, v in estimator.get_params(deep=False).items():
        if len(out) >= max_keys:
            break
        if isinstance(v, (bool, int, float)):
            out[k] = v
        elif isinstance(v, str) and len(v) < 200:
            out[k] = v
        elif v is None:
            out[k] = "null"
    return out


def log_compare_models_to_mlflow(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    tracking_uri: str | Path,
    experiment_name: str,
    dataset_sha256: str | None = None,
    extra_params: dict[str, Any] | None = None,
    run_name_prefix: str = "",
    log_training_curves: bool = True,
    log_sklearn_models: bool = True,
    log_mlp_torch: bool = True,
    **compare_kwargs: Any,
) -> pd.DataFrame:
    """
    Executa `compare_models_holdout` e cria **um run por modelo** com parâmetros e métricas.

    Por padrão grava também **modelos** no store: `mlflow.sklearn.log_model` em cada baseline
    e `mlflow.pytorch.log_model` no run `churn_mlp` (Etapa 2 — artefatos reprodutíveis).
    Curva de loss da MLP continua em CSV sob `training/`. Use `False` em testes rápidos.
    """
    kwargs = dict(compare_kwargs)
    kwargs["return_val_artifacts"] = True
    table, art = compare_models_holdout(X, y, **kwargs)

    mlflow.set_tracking_uri(_normalize_tracking_uri(tracking_uri))
    mlflow.set_experiment(experiment_name)

    split_meta = art["split_meta"]
    common: dict[str, Any] = {
        "protocol": "stratified_holdout",
        "n_rows_total": len(X),
        "n_features_raw": int(X.shape[1]),
        **split_meta,
    }
    if dataset_sha256:
        common["dataset_sha256"] = dataset_sha256
    if extra_params:
        common.update(extra_params)

    fitted = art["fitted_sklearn"]
    mlp_train = art["mlp_train_out"]
    mlp_mod = art["mlp_model"]

    for model_name in table.index:
        run_name = f"{run_name_prefix}{model_name}" if run_name_prefix else model_name
        with mlflow.start_run(run_name=run_name):
            merged_common = {**common, "tracked_model": model_name}
            mlflow.log_params(_flatten_params("common", merged_common))

            if model_name in fitted:
                sk_params = _simple_estimator_params(fitted[model_name].named_steps["model"])
                if sk_params:
                    mlflow.log_params(_flatten_params("sklearn", sk_params))

            if model_name == "churn_mlp":
                cfg = kwargs.get("mlp_train_config") or TrainConfig()
                mlflow.log_params(_flatten_params("mlp_train", asdict(cfg)))
                mlflow.log_params(
                    _flatten_params(
                        "mlp_fit",
                        {
                            "epochs_run": mlp_train["epochs_run"],
                            "stopped_early": mlp_train["stopped_early"],
                            "device": mlp_train["device"],
                        },
                    )
                )
                mlflow.log_metric(
                    "mlp_best_val_loss",
                    float(min(mlp_train["history"]["val_loss"])),
                )

            for col in table.columns:
                mlflow.log_metric(col, float(table.loc[model_name, col]))

            if log_training_curves and model_name == "churn_mlp":
                hist = pd.DataFrame(mlp_train["history"])
                with tempfile.TemporaryDirectory() as td:
                    path = Path(td) / "mlp_training_losses.csv"
                    hist.to_csv(path, index=False)
                    mlflow.log_artifact(str(path), artifact_path="training")

            if log_sklearn_models and model_name in fitted:
                try:
                    sklearn_flavor = importlib.import_module("mlflow.sklearn")
                    sklearn_flavor.log_model(fitted[model_name], artifact_path="sklearn_model")
                except Exception as e:
                    logger.warning("mlflow.sklearn.log_model falhou: %s", e)

            if log_mlp_torch and model_name == "churn_mlp":
                try:
                    pytorch_flavor = importlib.import_module("mlflow.pytorch")
                    pytorch_flavor.log_model(mlp_mod.cpu(), artifact_path="torch_mlp")
                except Exception as e:
                    logger.warning("mlflow.pytorch.log_model falhou: %s", e)

            if model_name == "churn_mlp":
                with tempfile.TemporaryDirectory() as td:
                    fp = Path(td) / "comparison_all_models_metrics.csv"
                    table.to_csv(fp)
                    mlflow.log_artifact(str(fp), artifact_path="summary")

    return table
