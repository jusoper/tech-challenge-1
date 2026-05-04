"""Transformadores sklearn customizados (Etapa 3 — tarefa 2)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

from telco_churn.data.preprocessing import build_feature_preprocessor, infer_column_types


class TelcoSklearnFeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Descobre colunas numéricas vs categóricas (`infer_column_types`) e ajusta um
    `ColumnTransformer` com imputação + scaler / OHE — encapsula o pré-processamento
    usado nos baselines e na MLP.
    """

    encoder_: ColumnTransformer

    def fit(self, X: Any, y: Any | None = None) -> TelcoSklearnFeatureEncoder:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TelcoSklearnFeatureEncoder espera pandas.DataFrame")
        self.feature_names_in_ = X.columns.to_numpy(dtype=object)
        self.n_features_in_ = X.shape[1]
        num_cols, cat_cols = infer_column_types(X.columns)
        self.encoder_ = build_feature_preprocessor(num_cols, cat_cols)
        self.encoder_.fit(X, y)
        return self

    def transform(self, X: Any) -> Any:
        return self.encoder_.transform(X)

    def get_feature_names_out(self, input_features: Any | None = None) -> Any:
        return self.encoder_.get_feature_names_out()
