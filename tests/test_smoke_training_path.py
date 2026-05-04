"""Smoke test do caminho principal de dados + modelo (Etapa 3 — tarefa 3)."""

import numpy as np
import pandas as pd

from telco_churn import (
    TrainConfig,
    build_telco_feature_transform_pipeline,
    compare_models_holdout,
    prepare_telco_features,
)


def test_smoke_import_package_version() -> None:
    import telco_churn

    assert telco_churn.__version__
    assert hasattr(telco_churn, "validate_telco_raw_supervised")


def test_smoke_prepare_fit_feature_pipeline_and_mini_compare() -> None:
    rng = np.random.default_rng(7)
    n = 120
    df = pd.DataFrame(
        {
            "tenure": rng.integers(0, 48, size=n),
            "MonthlyCharges": rng.normal(65.0, 20.0, size=n).clip(20.0, 110.0),
            "TotalCharges": rng.normal(2000, 400, size=n).clip(0, None),
            "gender": rng.choice(["Male", "Female"], size=n),
            "Partner": rng.choice(["Yes", "No"], size=n),
            "Churn": rng.binomial(1, 0.28, size=n),
        }
    )
    X, y = prepare_telco_features(df)
    pipe = build_telco_feature_transform_pipeline()
    Z = pipe.fit_transform(X, y)
    assert Z.shape[0] == n

    table = compare_models_holdout(
        X,
        y,
        random_state=0,
        test_size=0.3,
        mlp_hidden_dims=(8, 4),
        mlp_train_config=TrainConfig(
            batch_size=32,
            max_epochs=4,
            patience=2,
            learning_rate=0.05,
            seed=0,
        ),
        device="cpu",
        return_val_artifacts=False,
    )
    assert table.shape[0] == 5
    assert "roc_auc" in table.columns
