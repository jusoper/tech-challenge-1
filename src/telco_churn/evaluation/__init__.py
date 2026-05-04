"""Avaliação em holdout e métricas (separado de dados, treino e tracking)."""

from telco_churn.evaluation.baselines import default_churn_sklearn_models
from telco_churn.evaluation.holdout import compare_models_holdout
from telco_churn.evaluation.metrics import compute_binary_metrics

__all__ = [
    "compare_models_holdout",
    "compute_binary_metrics",
    "default_churn_sklearn_models",
]
