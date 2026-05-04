"""Classificadores sklearn usados como baseline no holdout (baixo acoplamento à avaliação)."""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def default_churn_sklearn_models(random_state: int) -> list[tuple[str, Any]]:
    """Ordem estável: dummy estratificado, logística balanceada, RF, HGB."""
    return [
        ("dummy_stratified", DummyClassifier(strategy="stratified", random_state=random_state)),
        (
            "logistic_regression_balanced",
            LogisticRegression(
                max_iter=2000,
                random_state=random_state,
                class_weight="balanced",
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=80,
                max_depth=10,
                random_state=random_state,
                class_weight="balanced",
                n_jobs=1,
            ),
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_iter=120,
                max_depth=6,
                random_state=random_state,
                class_weight="balanced",
            ),
        ),
    ]
