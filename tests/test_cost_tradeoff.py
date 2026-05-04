"""Trade-off custo FP vs FN e limiar ótimo."""

import numpy as np

from telco_churn.business.cost_tradeoff import (
    business_value_proxy,
    compare_thresholds_report,
    confusion_counts,
    costs_at_threshold,
    misclassification_cost,
    optimal_threshold_min_cost,
    sweep_threshold_costs,
)


def test_confusion_counts_and_misclassification_cost() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    assert (tn, fp, fn, tp) == (1, 1, 1, 1)
    assert misclassification_cost(fp, fn, cost_fp=10.0, cost_fn=100.0) == 10.0 + 100.0


def test_costs_at_threshold() -> None:
    y = np.array([0, 1])
    s = np.array([0.2, 0.8])
    r = costs_at_threshold(y, s, 0.5, cost_fp=1.0, cost_fn=10.0)
    assert r["fp"] == 0 and r["fn"] == 0 and r["total_cost"] == 0.0


def test_optimal_threshold_reduces_cost_vs_mid() -> None:
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, size=n)
    # scores correlacionados ao alvo (modelo razoável)
    s = np.where(y == 1, rng.uniform(0.35, 0.95, size=n), rng.uniform(0.05, 0.65, size=n))
    c_fp, c_fn = 50.0, 1200.0
    t_opt, _ = optimal_threshold_min_cost(y, s, cost_fp=c_fp, cost_fn=c_fn)
    base = costs_at_threshold(y, s, 0.5, cost_fp=c_fp, cost_fn=c_fn)
    opt = costs_at_threshold(y, s, t_opt, cost_fp=c_fp, cost_fn=c_fn)
    assert opt["total_cost"] <= base["total_cost"]


def test_sweep_threshold_costs_shape() -> None:
    y = np.array([0, 1, 0, 1, 0])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.15])
    df = sweep_threshold_costs(y, s, cost_fp=1.0, cost_fn=2.0)
    assert len(df) == 101
    assert set(df.columns) == {"threshold", "tn", "fp", "fn", "tp", "total_cost"}


def test_business_value_proxy_matches_notebook_semantics() -> None:
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 1, 0])
    # tp=1, fp=1 -> 1200 - 50
    v = business_value_proxy(y_true, y_pred, ltov=1200.0, cost_action=50.0)
    assert v == 1200.0 - 50.0


def test_compare_thresholds_report_two_rows() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=200)
    s = rng.uniform(0, 1, size=200)
    rep = compare_thresholds_report(y, s, baseline_threshold=0.5, cost_fp=50.0, cost_fn=1200.0)
    assert len(rep) == 2
    assert "default_threshold" in rep.index
    assert "min_total_cost" in rep.index
