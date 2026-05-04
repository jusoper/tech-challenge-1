"""Comparação MLP vs baselines e métricas binárias."""

import numpy as np
import pandas as pd

from telco_churn import (
    TrainConfig,
    compare_models_holdout,
    compute_binary_metrics,
    prepare_telco_features,
)


def test_compute_binary_metrics_four_metrics() -> None:
    y_true = np.array([0, 0, 1, 1, 1])
    # scores ordenados: melhor ROC / PR
    y_score = np.array([0.1, 0.2, 0.55, 0.7, 0.9])
    m = compute_binary_metrics(y_true, y_score, threshold=0.5)
    assert set(m.keys()) == {"roc_auc", "pr_auc", "f1", "accuracy"}
    assert all(np.isfinite(list(m.values())))


def _toy_telco_like(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, size=n)
    monthly = rng.normal(65.0, 25.0, size=n).clip(18.0, 120.0)
    total = (monthly * tenure).clip(0, 9000) + rng.normal(0, 200, size=n)
    churn_p = 1 / (1 + np.exp(-0.02 * (monthly - 70) - 0.01 * (tenure - 24)))
    churn = rng.binomial(1, churn_p)
    return pd.DataFrame(
        {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total.clip(0, None),
            "gender": rng.choice(["Male", "Female"], size=n),
            "Partner": rng.choice(["Yes", "No"], size=n),
            "PhoneService": rng.choice(["Yes", "No"], size=n),
            "Churn": churn,
        }
    )


def test_compare_models_holdout_includes_mlp_and_sklearn() -> None:
    df = _toy_telco_like()
    X, y = prepare_telco_features(df)
    table, art = compare_models_holdout(
        X,
        y,
        random_state=42,
        test_size=0.25,
        mlp_hidden_dims=(24, 12),
        mlp_train_config=TrainConfig(
            batch_size=64,
            max_epochs=25,
            patience=8,
            learning_rate=0.02,
            seed=42,
        ),
        device="cpu",
        return_val_artifacts=True,
    )
    assert "y_val" in art and "scores" in art
    assert "churn_mlp" in art["scores"]
    assert "mlp_train_out" in art and "mlp_model" in art
    assert "fitted_sklearn" in art and "split_meta" in art
    assert table.shape[0] == 5
    assert table.shape[1] == 4
    expected_models = {
        "dummy_stratified",
        "logistic_regression_balanced",
        "random_forest",
        "hist_gradient_boosting",
        "churn_mlp",
    }
    assert set(table.index) == expected_models
    for col in ["roc_auc", "pr_auc", "f1", "accuracy"]:
        assert col in table.columns
        assert table[col].notna().all()


def test_prepare_telco_features_yes_no_churn() -> None:
    df = pd.DataFrame(
        {
            "tenure": [1, 12],
            "MonthlyCharges": [50.0, 70.0],
            "TotalCharges": ["50", "840"],
            "gender": ["Male", "Female"],
            "Churn": ["No", "Yes"],
        }
    )
    X, y = prepare_telco_features(df)
    assert list(y) == [0, 1]
    assert "Churn" not in X.columns
