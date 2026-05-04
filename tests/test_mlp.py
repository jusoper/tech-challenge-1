"""Testes mínimos da MLP (forma do tensor e loss)."""

import pytest
import torch

from telco_churn.mlp import ChurnMLP, churn_binary_loss


@pytest.mark.parametrize("activation", ["relu", "gelu", "tanh"])
def test_churn_mlp_forward_shape(activation: str) -> None:
    batch, input_dim = 16, 45
    model = ChurnMLP(
        input_dim=input_dim,
        hidden_dims=(32, 16),
        dropout=0.0,
        activation=activation,
    )
    x = torch.randn(batch, input_dim)
    logits = model(x)
    assert logits.shape == (batch, 1)


def test_churn_binary_loss_runs() -> None:
    model = ChurnMLP(input_dim=10, hidden_dims=(8,), dropout=0.0)
    crit = churn_binary_loss()
    x = torch.randn(4, 10)
    y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    loss = crit(model(x), y)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_invalid_activation_raises() -> None:
    with pytest.raises(ValueError, match="activation"):
        ChurnMLP(input_dim=5, hidden_dims=(4,), activation="sigmoid")
