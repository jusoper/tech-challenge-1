"""Avaliação em holdout e métricas (separado de dados, treino e tracking)."""

from telco_churn.evaluation.baselines import default_churn_sklearn_models
from telco_churn.evaluation.holdout import compare_models_holdout
from telco_churn.evaluation.metrics import compute_binary_metrics
from telco_churn.evaluation.stratified_cv import (
    compare_models_stratified_cv,
    compare_sklearn_baselines_stratified_cv,
    make_stratified_kfold,
    stratified_cv_meta,
)

__all__ = [
    "compare_models_holdout",
    "compare_models_stratified_cv",
    "compare_sklearn_baselines_stratified_cv",
    "compute_binary_metrics",
    "default_churn_sklearn_models",
    "make_stratified_kfold",
    "stratified_cv_meta",
]
