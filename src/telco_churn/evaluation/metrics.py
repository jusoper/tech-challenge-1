"""Métricas de classificação binária no conjunto de validação (Etapa 2 — tarefa 3)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score_positive: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Métricas em conjunto de validação: ROC-AUC, PR-AUC, F1, acurácia (≥4).
    `y_score_positive` é P(y=1) ou score monotônico equivalente.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score_positive, dtype=float).ravel()
    y_pred = (y_score >= threshold).astype(int)
    out: dict[str, float] = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    return out
