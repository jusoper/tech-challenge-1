"""Pacote do Tech Challenge — churn Telco (Etapa 3: módulos por domínio em `src/`)."""

from telco_churn.business.cost_tradeoff import (
    DEFAULT_COST_FN,
    DEFAULT_COST_FP,
    business_value_proxy,
    compare_thresholds_report,
    costs_at_threshold,
    optimal_threshold_min_cost,
    sweep_threshold_costs,
)
from telco_churn.data.pipeline import (
    build_telco_classifier_pipeline,
    build_telco_feature_transform_pipeline,
)
from telco_churn.data.preprocessing import TelcoTableSanitizer, prepare_telco_features
from telco_churn.data.transformers import TelcoSklearnFeatureEncoder
from telco_churn.evaluation.holdout import compare_models_holdout
from telco_churn.evaluation.metrics import compute_binary_metrics
from telco_churn.evaluation.stratified_cv import (
    compare_models_stratified_cv,
    compare_sklearn_baselines_stratified_cv,
    make_stratified_kfold,
    stratified_cv_meta,
)
from telco_churn.modeling.mlp import ChurnMLP, churn_binary_loss
from telco_churn.tracking.mlflow_compare import log_compare_models_to_mlflow
from telco_churn.training.train_mlp import EarlyStopping, TrainConfig, train_churn_mlp
from telco_churn.validation import (
    validate_telco_feature_matrix,
    validate_telco_raw_supervised,
)

__all__ = [
    "ChurnMLP",
    "DEFAULT_COST_FN",
    "DEFAULT_COST_FP",
    "EarlyStopping",
    "TelcoSklearnFeatureEncoder",
    "TelcoTableSanitizer",
    "TrainConfig",
    "business_value_proxy",
    "build_telco_classifier_pipeline",
    "build_telco_feature_transform_pipeline",
    "churn_binary_loss",
    "compare_models_holdout",
    "compare_models_stratified_cv",
    "compare_sklearn_baselines_stratified_cv",
    "make_stratified_kfold",
    "stratified_cv_meta",
    "compare_thresholds_report",
    "compute_binary_metrics",
    "log_compare_models_to_mlflow",
    "costs_at_threshold",
    "optimal_threshold_min_cost",
    "prepare_telco_features",
    "sweep_threshold_costs",
    "train_churn_mlp",
    "validate_telco_feature_matrix",
    "validate_telco_raw_supervised",
    "__version__",
]
__version__ = "0.1.0"
