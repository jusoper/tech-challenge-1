"""Carga e pré-processamento tabular (coesão: apenas dados)."""

from telco_churn.data.preprocessing import (
    NUMERIC_FEATURES,
    build_feature_preprocessor,
    infer_column_types,
    prepare_telco_features,
)

__all__ = [
    "NUMERIC_FEATURES",
    "build_feature_preprocessor",
    "infer_column_types",
    "prepare_telco_features",
]
