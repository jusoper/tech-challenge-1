"""Análise de trade-off de custo FP vs FN e escolha de threshold (Etapa 2 — tarefa 4)."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mesmas premissas ilustrativas do notebook de EDA (`LTV_MEDIO`, `CUSTO_ACAO`).
DEFAULT_COST_FN: float = 1200.0  # custo de não detectar churn (ex.: LTV não retida)
DEFAULT_COST_FP: float = 50.0  # custo de intervenção em quem não churnaria


def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[int, int, int, int]:
    """Retorna (TN, FP, FN, TP)."""
    yt = np.asarray(y_true).astype(int).ravel()
    yp = np.asarray(y_pred).astype(int).ravel()
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tp = int(np.sum((yt == 1) & (yp == 1)))
    return tn, fp, fn, tp


def misclassification_cost(fp: int, fn: int, cost_fp: float, cost_fn: float) -> float:
    """Custo total de erros: FP × custo_contato + FN × custo_perda_churn."""
    return float(fp) * float(cost_fp) + float(fn) * float(cost_fn)


def costs_at_threshold(
    y_true: np.ndarray,
    y_score_positive: np.ndarray,
    threshold: float,
    *,
    cost_fp: float = DEFAULT_COST_FP,
    cost_fn: float = DEFAULT_COST_FN,
) -> dict[str, float | int]:
    """
    Classifica com `y_score >= threshold` e devolve FP, FN e custo total
    (trade-off central entre alertas indevidos e churns perdidos).
    """
    yt = np.asarray(y_true).astype(int).ravel()
    ys = np.asarray(y_score_positive, dtype=float).ravel()
    y_pred = (ys >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_counts(yt, y_pred)
    total = misclassification_cost(fp, fn, cost_fp, cost_fn)
    return {
        "threshold": float(threshold),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "total_cost": float(total),
    }


def sweep_threshold_costs(
    y_true: np.ndarray,
    y_score_positive: np.ndarray,
    *,
    cost_fp: float = DEFAULT_COST_FP,
    cost_fn: float = DEFAULT_COST_FN,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """
    Varre limiares (padrão: grade 0..1) e devolve FP, FN e custo total por linha.
    Útil para visualizar o trade-off e localizar regiões operacionais.
    """
    yt = np.asarray(y_true).astype(int).ravel()
    ys = np.asarray(y_score_positive, dtype=float).ravel()
    if thresholds is None:
        th = np.linspace(0.0, 1.0, 101)
    else:
        th = np.asarray(list(thresholds), dtype=float)
    rows: list[dict[str, float | int]] = []
    for t in th:
        rows.append(costs_at_threshold(yt, ys, float(t), cost_fp=cost_fp, cost_fn=cost_fn))
    return pd.DataFrame(rows)


def optimal_threshold_min_cost(
    y_true: np.ndarray,
    y_score_positive: np.ndarray,
    *,
    cost_fp: float = DEFAULT_COST_FP,
    cost_fn: float = DEFAULT_COST_FN,
    thresholds: Iterable[float] | None = None,
) -> tuple[float, pd.Series]:
    """Limiar que minimiza `total_cost` na grade; retorna (threshold, linha do DataFrame)."""
    df = sweep_threshold_costs(
        y_true,
        y_score_positive,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
        thresholds=thresholds,
    )
    i = int(df["total_cost"].idxmin())
    row = df.loc[i]
    t_best = float(row["threshold"])
    logger.debug(
        "optimal_threshold=%s total_cost=%s fp=%s fn=%s",
        t_best,
        row["total_cost"],
        row["fp"],
        row["fn"],
    )
    return t_best, row


def business_value_proxy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    ltov: float = DEFAULT_COST_FN,
    cost_action: float = DEFAULT_COST_FP,
) -> float:
    """
    Proxy do notebook: soma TP×LTV − FP×custo de ação (sem termo explícito em FN).
    Complementa a visão de custo FP+FN para discussão com stakeholders.
    """
    yt = np.asarray(y_true).astype(int).ravel()
    yp = np.asarray(y_pred).astype(int).ravel()
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    return float(tp * ltov - fp * cost_action)


def compare_thresholds_report(
    y_true: np.ndarray,
    y_score_positive: np.ndarray,
    *,
    baseline_threshold: float = 0.5,
    cost_fp: float = DEFAULT_COST_FP,
    cost_fn: float = DEFAULT_COST_FN,
) -> pd.DataFrame:
    """
    Tabela-resumo: limiar fixo (ex.: 0,5) vs limiar de custo mínimo, com FP/FN e custos.
    """
    yt = np.asarray(y_true).astype(int).ravel()
    ys = np.asarray(y_score_positive, dtype=float).ravel()
    t_opt, row_opt = optimal_threshold_min_cost(yt, ys, cost_fp=cost_fp, cost_fn=cost_fn)
    base = costs_at_threshold(yt, ys, baseline_threshold, cost_fp=cost_fp, cost_fn=cost_fn)
    opt = row_opt.to_dict()
    return pd.DataFrame(
        [
            {"policy": "default_threshold", **base},
            {"policy": "min_total_cost", **opt},
        ]
    ).set_index("policy")
