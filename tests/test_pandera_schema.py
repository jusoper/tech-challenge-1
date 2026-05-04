"""Testes de schema Pandera (Etapa 3 — tarefa 3)."""

import pandas as pd
import pandera.errors
import pytest

from telco_churn.data.preprocessing import prepare_telco_features
from telco_churn.validation import validate_telco_feature_matrix, validate_telco_raw_supervised


def test_validate_raw_accepts_yes_no_churn() -> None:
    df = pd.DataFrame(
        {
            "tenure": [1, 12],
            "MonthlyCharges": [50.0, 70.0],
            "TotalCharges": ["50", "840"],
            "gender": ["Male", "Female"],
            "Churn": ["No", "Yes"],
        }
    )
    out = validate_telco_raw_supervised(df)
    assert len(out) == 2


def test_validate_raw_accepts_numeric_churn() -> None:
    df = pd.DataFrame(
        {
            "tenure": [0, 72],
            "MonthlyCharges": [18.0, 99.0],
            "Churn": [0, 1],
        }
    )
    validate_telco_raw_supervised(df)


def test_validate_raw_rejects_invalid_churn() -> None:
    df = pd.DataFrame(
        {
            "tenure": [1],
            "MonthlyCharges": [50.0],
            "Churn": ["Maybe"],
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        validate_telco_raw_supervised(df)


def test_validate_raw_rejects_negative_tenure() -> None:
    df = pd.DataFrame(
        {
            "tenure": [-1],
            "MonthlyCharges": [50.0],
            "Churn": [0],
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        validate_telco_raw_supervised(df)


def test_validate_feature_matrix_after_prepare() -> None:
    df = pd.DataFrame(
        {
            "tenure": [1, 24],
            "MonthlyCharges": [50.0, 80.0],
            "TotalCharges": ["50", "1920"],
            "x": ["a", "b"],
            "Churn": [0, 1],
        }
    )
    X, _ = prepare_telco_features(df)
    validate_telco_feature_matrix(X)


def test_validate_feature_matrix_rejects_negative_monthly() -> None:
    df = pd.DataFrame(
        {
            "tenure": [1.0],
            "MonthlyCharges": [-1.0],
            "TotalCharges": [10.0],
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        validate_telco_feature_matrix(df)
