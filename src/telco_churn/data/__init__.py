"""Carga e pré-processamento tabular (coesão: apenas dados)."""

from telco_churn.data.pipeline import (
    build_telco_classifier_pipeline,
    build_telco_feature_transform_pipeline,
)
from telco_churn.data.preprocessing import (
    NUMERIC_FEATURES,
    TelcoTableSanitizer,
    build_feature_preprocessor,
    infer_column_types,
    prepare_telco_features,
)
from telco_churn.data.transformers import TelcoSklearnFeatureEncoder

__all__ = [
    "NUMERIC_FEATURES",
    "TelcoSklearnFeatureEncoder",
    "TelcoTableSanitizer",
    "build_feature_preprocessor",
    "build_telco_classifier_pipeline",
    "build_telco_feature_transform_pipeline",
    "infer_column_types",
    "prepare_telco_features",
]
