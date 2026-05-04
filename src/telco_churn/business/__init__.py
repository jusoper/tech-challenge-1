"""Regras de negócio e custo operacional (FP/FN), desacoplado do sklearn/PyTorch."""

from telco_churn.business.cost_tradeoff import (
    DEFAULT_COST_FN,
    DEFAULT_COST_FP,
    business_value_proxy,
    compare_thresholds_report,
    confusion_counts,
    costs_at_threshold,
    misclassification_cost,
    optimal_threshold_min_cost,
    sweep_threshold_costs,
)

__all__ = [
    "DEFAULT_COST_FN",
    "DEFAULT_COST_FP",
    "business_value_proxy",
    "compare_thresholds_report",
    "confusion_counts",
    "costs_at_threshold",
    "misclassification_cost",
    "optimal_threshold_min_cost",
    "sweep_threshold_costs",
]
