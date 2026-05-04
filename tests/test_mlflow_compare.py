"""MLflow: um run por modelo na comparação holdout."""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from telco_churn.mlflow_compare import log_compare_models_to_mlflow
from telco_churn.preprocessing import prepare_telco_features
from telco_churn.train_mlp import TrainConfig


def _tiny_churn_df(n: int = 200, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 60, size=n)
    monthly = rng.normal(60.0, 20.0, size=n).clip(20.0, 100.0)
    total = (monthly * tenure).clip(0, 8000) + rng.normal(0, 100, size=n)
    churn_p = 1 / (1 + np.exp(-0.03 * (monthly - 65)))
    churn = rng.binomial(1, churn_p)
    return pd.DataFrame(
        {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total.clip(0, None),
            "gender": rng.choice(["Male", "Female"], size=n),
            "Partner": rng.choice(["Yes", "No"], size=n),
            "Churn": churn,
        }
    )


def test_log_compare_models_to_mlflow_file_store(tmp_path: Path) -> None:
    uri = f"file:{tmp_path / 'mlruns'}"
    df = _tiny_churn_df()
    X, y = prepare_telco_features(df)
    table = log_compare_models_to_mlflow(
        X,
        y,
        tracking_uri=uri,
        experiment_name="pytest-etapa2-compare",
        dataset_sha256="abc123",
        extra_params={"source": "unit_test"},
        run_name_prefix="t-",
        log_training_curves=True,
        log_sklearn_models=False,
        log_mlp_torch=False,
        random_state=42,
        test_size=0.25,
        mlp_hidden_dims=(16, 8),
        mlp_train_config=TrainConfig(
            batch_size=32,
            max_epochs=6,
            patience=4,
            learning_rate=0.02,
            seed=1,
        ),
        device="cpu",
    )
    assert table.shape[0] == 5

    exp = mlflow.get_experiment_by_name("pytest-etapa2-compare")
    assert exp is not None
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 5
