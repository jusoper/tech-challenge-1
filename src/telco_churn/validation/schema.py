"""Contratos de dados com Pandera (Etapa 3 — tarefa 3: testes de schema)."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa


def _churn_labels_ok(series: pd.Series) -> bool:
    """Aceita Churn binário (0/1) ou strings Yes/No (dataset IBM Telco)."""
    s = series.dropna()
    if len(s) == 0:
        return False
    return bool(s.astype(str).str.strip().isin(["Yes", "No", "0", "1"]).all())


telco_raw_supervised_schema = pa.DataFrameSchema(
    {
        "Churn": pa.Column(
            str,
            nullable=False,
            coerce=True,
            checks=pa.Check(_churn_labels_ok, element_wise=False, error="Churn inválido"),
        ),
        "tenure": pa.Column(
            float,
            nullable=False,
            coerce=True,
            checks=pa.Check.ge(0, error="tenure deve ser >= 0"),
        ),
        "MonthlyCharges": pa.Column(
            float,
            nullable=False,
            coerce=True,
            checks=pa.Check.ge(0, error="MonthlyCharges deve ser >= 0"),
        ),
    },
    strict=False,
)


telco_feature_matrix_schema = pa.DataFrameSchema(
    {
        "tenure": pa.Column(
            float,
            nullable=True,
            coerce=True,
            checks=pa.Check.ge(0, error="tenure deve ser >= 0"),
        ),
        "MonthlyCharges": pa.Column(
            float,
            nullable=False,
            coerce=True,
            checks=pa.Check.ge(0, error="MonthlyCharges deve ser >= 0"),
        ),
        "TotalCharges": pa.Column(float, nullable=True, coerce=True),
    },
    strict=False,
)


def validate_telco_raw_supervised(df: pd.DataFrame) -> pd.DataFrame:
    """Valida entrada típica com alvo `Churn` antes de `prepare_telco_features`."""
    return telco_raw_supervised_schema.validate(df)


def validate_telco_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Valida matriz de features (p.ex. após `prepare_telco_features`)."""
    return telco_feature_matrix_schema.validate(df)
