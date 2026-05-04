"""Loop de treinamento da MLP com batching e early stopping (Etapa 2 — tarefa 2)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from telco_churn.modeling.mlp import ChurnMLP, churn_binary_loss

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    max_epochs: int = 500
    patience: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    min_delta: float = 0.0
    seed: int = 42


class EarlyStopping:
    """Interrompe quando a métrica de validação não melhora por `patience` épocas seguidas."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        if patience < 1:
            raise ValueError("patience deve ser >= 1")
        self.patience = patience
        self.min_delta = float(min_delta)
        self.best: float | None = None
        self.bad_epochs = 0
        self.best_state: dict[str, torch.Tensor] | None = None

    def is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        return value < self.best - self.min_delta

    def step(self, value: float, model: nn.Module) -> bool:
        """Atualiza estado; retorna True se deve parar o treino."""
        if self.is_improvement(value):
            self.best = value
            self.bad_epochs = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def load_best(self, model: nn.Module, device: torch.device) -> None:
        if self.best_state is None:
            return
        model.load_state_dict({k: v.to(device) for k, v in self.best_state.items()})


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_xy_tensors(
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if X.ndim != 2:
        raise ValueError("X deve ser 2D (n_samples, n_features)")
    y = np.asarray(y).reshape(-1, 1).astype(np.float32)
    x_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
    return x_t, y_t


def _epoch_train(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total, n_batches = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        n_batches += 1
    return total / max(n_batches, 1)


@torch.no_grad()
def _epoch_eval(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total, n_batches = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        total += float(criterion(model(xb), yb))
        n_batches += 1
    return total / max(n_batches, 1)


def train_churn_mlp(
    model: ChurnMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    config: TrainConfig | None = None,
    device: torch.device | str | None = None,
) -> dict[str, object]:
    """
    Treina com AdamW, batching e early stopping na loss de validação (BCE com logits).
    Ao final, restaura os pesos da melhor época de validação.
    """
    cfg = config or TrainConfig()
    _set_seed(cfg.seed)
    dev: torch.device
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    model = model.to(dev)
    criterion = churn_binary_loss()

    x_train, y_train_t = _to_xy_tensors(X_train, y_train, dev)
    x_val, y_val_t = _to_xy_tensors(X_val, y_val, dev)

    train_ds = TensorDataset(x_train.cpu(), y_train_t.cpu())
    val_ds = TensorDataset(x_val.cpu(), y_val_t.cpu())
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    early = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)
    history_train: list[float] = []
    history_val: list[float] = []
    stopped_early = False

    for epoch in range(cfg.max_epochs):
        train_loss = _epoch_train(model, train_loader, criterion, optimizer, dev)
        val_loss = _epoch_eval(model, val_loader, criterion, dev)
        history_train.append(train_loss)
        history_val.append(val_loss)
        logger.debug("epoch=%s train_loss=%.6f val_loss=%.6f", epoch + 1, train_loss, val_loss)

        if early.step(val_loss, model):
            stopped_early = True
            break

    early.load_best(model, dev)
    model.eval()

    return {
        "history": {"train_loss": history_train, "val_loss": history_val},
        "epochs_run": len(history_train),
        "stopped_early": stopped_early,
        "device": str(dev),
    }
