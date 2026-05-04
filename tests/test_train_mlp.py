"""Treino MLP: early stopping e smoke do loop com batching."""

import numpy as np
import torch

from telco_churn.mlp import ChurnMLP
from telco_churn.train_mlp import EarlyStopping, TrainConfig, train_churn_mlp


def test_early_stopping_patience() -> None:
    model = ChurnMLP(input_dim=3, hidden_dims=(4,), dropout=0.0)
    es = EarlyStopping(patience=2, min_delta=0.0)
    assert not es.step(1.0, model)
    assert not es.step(0.5, model)
    assert not es.step(0.6, model)
    assert es.step(0.7, model)


def test_early_stopping_min_delta() -> None:
    model = ChurnMLP(input_dim=2, hidden_dims=(3,), dropout=0.0)
    es = EarlyStopping(patience=2, min_delta=0.1)
    assert not es.step(1.0, model)
    assert not es.step(0.85, model)
    assert not es.step(0.92, model)
    assert es.step(0.90, model)


def test_train_churn_mlp_smoke_cpu() -> None:
    rng = np.random.default_rng(0)
    n, d = 120, 8
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (rng.standard_normal(n) > 0).astype(np.float32)
    split = 80
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = ChurnMLP(input_dim=d, hidden_dims=(16, 8), dropout=0.0)
    out = train_churn_mlp(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        config=TrainConfig(
            batch_size=32,
            max_epochs=5,
            patience=99,
            learning_rate=0.01,
            seed=1,
        ),
        device="cpu",
    )

    assert out["epochs_run"] == 5
    assert out["stopped_early"] is False
    hist = out["history"]
    assert len(hist["train_loss"]) == 5
    assert len(hist["val_loss"]) == 5
    assert all(np.isfinite(hist["train_loss"]))
    assert all(np.isfinite(hist["val_loss"]))

    xb = torch.as_tensor(X_val[:4], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(xb)
    assert logits.shape == (4, 1)
