"""Testes unitários de pré-processamento (Etapa 3 — tarefa 3)."""

import pytest

from telco_churn.data.preprocessing import infer_column_types


@pytest.mark.parametrize(
    ("columns", "expected_numeric", "expected_categorical"),
    [
        (
            ["tenure", "MonthlyCharges", "TotalCharges", "gender"],
            ["tenure", "MonthlyCharges", "TotalCharges"],
            ["gender"],
        ),
        (["gender", "Partner"], [], ["gender", "Partner"]),
        (["tenure"], ["tenure"], []),
        (["MonthlyCharges", "Contract"], ["MonthlyCharges"], ["Contract"]),
    ],
)
def test_infer_column_types_partition(
    columns: list[str],
    expected_numeric: list[str],
    expected_categorical: list[str],
) -> None:
    num, cat = infer_column_types(columns)
    assert num == expected_numeric
    assert cat == expected_categorical
