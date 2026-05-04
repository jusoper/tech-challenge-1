"""Pacote do Tech Challenge — churn Telco (estrutura Etapa 1+)."""

from telco_churn.mlp import ChurnMLP, churn_binary_loss
from telco_churn.train_mlp import EarlyStopping, TrainConfig, train_churn_mlp

__all__ = [
    "ChurnMLP",
    "EarlyStopping",
    "TrainConfig",
    "churn_binary_loss",
    "train_churn_mlp",
    "__version__",
]
__version__ = "0.1.0"
