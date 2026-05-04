"""Pacote do Tech Challenge — churn Telco (estrutura Etapa 1+)."""

from telco_churn.compare_models import compare_models_holdout, compute_binary_metrics
from telco_churn.cost_tradeoff import (
    DEFAULT_COST_FN,
    DEFAULT_COST_FP,
    business_value_proxy,
    compare_thresholds_report,
    costs_at_threshold,
    optimal_threshold_min_cost,
    sweep_threshold_costs,
)
from telco_churn.mlflow_compare import log_compare_models_to_mlflow
from telco_churn.mlp import ChurnMLP, churn_binary_loss
from telco_churn.preprocessing import prepare_telco_features
from telco_churn.train_mlp import EarlyStopping, TrainConfig, train_churn_mlp

__all__ = [
    "ChurnMLP",
    "DEFAULT_COST_FN",
    "DEFAULT_COST_FP",
    "EarlyStopping",
    "TrainConfig",
    "business_value_proxy",
    "churn_binary_loss",
    "compare_models_holdout",
    "compare_thresholds_report",
    "compute_binary_metrics",
    "log_compare_models_to_mlflow",
    "costs_at_threshold",
    "optimal_threshold_min_cost",
    "prepare_telco_features",
    "sweep_threshold_costs",
    "train_churn_mlp",
    "__version__",
]
__version__ = "0.1.0"
