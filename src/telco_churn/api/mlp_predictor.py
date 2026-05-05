"""MLP + pré-processamento Telco, interface tipo sklearn (`predict_proba`)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline

from telco_churn.modeling.mlp import ChurnMLP

logger = logging.getLogger(__name__)


class TelcoMlpPredictor:
    """
    Wrapper para servir `ChurnMLP` na API: mesmo contrato que `sklearn.Pipeline` em inferência.

    `prep` deve ser o pipeline **somente de features** (`build_telco_feature_transform_pipeline`),
    já ajustado (`fit`).
    """

    def __init__(
        self,
        prep: Pipeline,
        mlp: ChurnMLP,
        device: torch.device | str | None = None,
    ) -> None:
        if not isinstance(prep, Pipeline):
            raise TypeError("prep deve ser sklearn.Pipeline (features Telco)")
        self.prep = prep
        self.mlp = mlp
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.mlp.to(self.device)
        self.mlp.eval()

    def _features_matrix(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self.prep.transform(X)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        return np.asarray(Xt, dtype=np.float32)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna `(n_samples, 2)` como classificadores sklearn binários."""
        Xm = self._features_matrix(X)
        xt = torch.as_tensor(Xm, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.mlp(xt)
            # `.tolist()` evita dependência de Tensor→NumPy em builds PyTorch sem NumPy.
            p1 = np.asarray(
                torch.sigmoid(logits).detach().cpu().flatten().tolist(),
                dtype=np.float64,
            )
        p1 = np.clip(p1, 0.0, 1.0)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


def save_mlp_predictor(path: Union[str, Path], predictor: TelcoMlpPredictor) -> None:
    """Persiste bundle para `TELCO_MLP_BUNDLE_PATH` (via joblib)."""
    joblib.dump(predictor, path)


def load_mlp_predictor(path: Union[str, Path]) -> TelcoMlpPredictor:
    obj = joblib.load(path)
    if not isinstance(obj, TelcoMlpPredictor):
        raise TypeError(f"Arquivo deve conter TelcoMlpPredictor, obtido {type(obj)}")
    return obj
