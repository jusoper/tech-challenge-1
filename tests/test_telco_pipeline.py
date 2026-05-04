"""Pipeline sklearn reprodutível com transformadores Telco (Etapa 3 — tarefa 2)."""

import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from telco_churn import (
    build_telco_classifier_pipeline,
    build_telco_feature_transform_pipeline,
    prepare_telco_features,
)
from telco_churn.data.preprocessing import TelcoTableSanitizer


def test_telco_table_sanitizer_drops_id_and_coerces_total_charges() -> None:
    df = pd.DataFrame(
        {
            "customerID": ["a", "b"],
            "TotalCharges": ["12.5", ""],
            "tenure": [1, 2],
            "Churn": [0, 1],
        }
    )
    out = TelcoTableSanitizer().fit_transform(df)
    assert "customerID" not in out.columns
    assert pd.api.types.is_numeric_dtype(out["TotalCharges"])
    assert out["TotalCharges"].isna().iloc[1]


def test_feature_transform_pipeline_matches_manual_column_transformer() -> None:
    df = pd.DataFrame(
        {
            "tenure": [1, 12, 24],
            "MonthlyCharges": [50.0, 70.0, 80.0],
            "TotalCharges": ["50", "840", "1920"],
            "gender": ["Male", "Female", "Male"],
            "Churn": [0, 1, 0],
        }
    )
    X, y = prepare_telco_features(df)
    pipe = build_telco_feature_transform_pipeline()
    Z = pipe.fit_transform(X, y)
    assert isinstance(Z, np.ndarray)
    assert Z.shape[0] == len(X)
    assert Z.shape[1] >= 3


def test_classifier_pipeline_fit_predict_proba() -> None:
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame(
        {
            "tenure": rng.integers(0, 60, size=n),
            "MonthlyCharges": rng.normal(60.0, 15.0, size=n),
            "TotalCharges": rng.normal(2000, 500, size=n),
            "gender": rng.choice(["Male", "Female"], size=n),
            "Churn": rng.integers(0, 2, size=n),
        }
    )
    X, y = prepare_telco_features(df)
    pipe = build_telco_classifier_pipeline(
        LogisticRegression(max_iter=500, random_state=0),
    )
    pipe.fit(X, y)
    p = pipe.predict_proba(X)[:, 1]
    assert p.shape == (n,)
    assert np.isfinite(p).all()


def test_feature_pipeline_pickle_roundtrip() -> None:
    df = pd.DataFrame(
        {
            "tenure": [1, 2],
            "MonthlyCharges": [50.0, 60.0],
            "TotalCharges": ["50", "120"],
            "x": ["a", "b"],
            "Churn": [0, 1],
        }
    )
    X, y = prepare_telco_features(df)
    pipe = build_telco_feature_transform_pipeline()
    pipe.fit(X, y)
    blob = pickle.dumps(pipe)
    restored = pickle.loads(blob)
    np.testing.assert_array_almost_equal(restored.transform(X), pipe.transform(X))
