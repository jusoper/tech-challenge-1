"""Pipelines sklearn reprodutíveis (Etapa 3 — tarefa 2): Telco + encoding + classificador."""

from __future__ import annotations

from typing import Any

from sklearn.base import clone
from sklearn.pipeline import Pipeline

from telco_churn.data.preprocessing import TelcoTableSanitizer
from telco_churn.data.transformers import TelcoSklearnFeatureEncoder


def build_telco_feature_transform_pipeline(*, transform_output: str | None = None) -> Pipeline:
    """
    Pipeline só de features: limpeza de domínio + `ColumnTransformer` (num/cat).

    Com `transform_output="pandas"` (sklearn ≥ 1.2), tenta manter `DataFrame` entre passos.
    """
    pipe = Pipeline(
        steps=[
            ("telco_sanitize", TelcoTableSanitizer()),
            ("telco_encode", TelcoSklearnFeatureEncoder()),
        ]
    )
    if transform_output is not None and hasattr(pipe, "set_output"):
        pipe.set_output(transform=transform_output)
    return pipe


def build_telco_classifier_pipeline(
    estimator: Any,
    *,
    transform_output: str | None = None,
) -> Pipeline:
    """Pipeline supervisionado completo para classificação binária de churn (features + modelo)."""
    pipe = Pipeline(
        steps=[
            ("telco_sanitize", TelcoTableSanitizer()),
            ("telco_encode", TelcoSklearnFeatureEncoder()),
            ("model", clone(estimator)),
        ]
    )
    if transform_output is not None and hasattr(pipe, "set_output"):
        pipe.set_output(transform=transform_output)
    return pipe
