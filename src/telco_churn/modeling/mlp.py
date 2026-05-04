"""MLP em PyTorch para churn binário (Etapa 2 — tarefa 1: arquitetura, ativação, loss)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class ChurnMLP(nn.Module):
    """MLP tabular: vetor de features → um logit (classe positiva de churn)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        if input_dim < 1:
            raise ValueError("input_dim deve ser >= 1")
        if not hidden_dims:
            raise ValueError("hidden_dims não pode ser vazio")
        act_key = activation.lower()
        if act_key not in _ACTIVATIONS:
            allowed = ", ".join(sorted(_ACTIVATIONS))
            raise ValueError(f"activation deve ser um de: {allowed}")
        super().__init__()
        act_cls = _ACTIVATIONS[act_key]
        dims = [input_dim, *hidden_dims, 1]
        blocks: list[nn.Module] = []
        for i in range(len(dims) - 1):
            blocks.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                blocks.append(act_cls())
                if dropout and dropout > 0:
                    blocks.append(nn.Dropout(p=float(dropout)))
        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def churn_binary_loss() -> nn.BCEWithLogitsLoss:
    """Loss para alvo binário {0,1} com saída em logit (sem sigmoid na última camada)."""
    return nn.BCEWithLogitsLoss()
