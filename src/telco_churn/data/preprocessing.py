"""Pré-processamento tabular alinhado ao notebook de EDA / baselines (Etapa 1)."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES: tuple[str, ...] = ("tenure", "MonthlyCharges", "TotalCharges")


def infer_column_types(feature_columns: Iterable[str]) -> tuple[list[str], list[str]]:
    """Colunas numéricas Telco conhecidas + demais como categóricas."""
    cols = list(feature_columns)
    numeric = [c for c in NUMERIC_FEATURES if c in cols]
    categorical = [c for c in cols if c not in numeric]
    return numeric, categorical


def build_feature_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """ColumnTransformer com imputação + scaler (num) e OHE (cat)."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))
    if not transformers:
        raise ValueError("numeric_cols e categorical_cols não podem ser ambos vazios")
    return ColumnTransformer(transformers=transformers)


def prepare_telco_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Remove `customerID`, binariza `Churn` e devolve (X, y)."""
    out = df.copy()
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    if "customerID" in out.columns:
        out = out.drop(columns=["customerID"])
    if "Churn" not in out.columns:
        raise ValueError("DataFrame deve conter coluna 'Churn'")
    if out["Churn"].dtype == object:
        out["Churn"] = (out["Churn"] == "Yes").astype(int)
    y = out["Churn"].astype(int)
    X = out.drop(columns=["Churn"])
    return X, y
