"""Contratos HTTP Pydantic para inferência (Etapa 3 — tarefa 4)."""

from __future__ import annotations

from typing import Any, Optional, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TelcoInferenceRow(BaseModel):
    """
    Uma linha de cliente Telco para inferência.

    Campos numéricos centrais são obrigatórios; demais colunas do IBM Telco podem
    ser enviadas como campos extras (`model_config.extra="allow"`).
    """

    model_config = ConfigDict(extra="allow")

    tenure: float = Field(ge=0, description="Meses de permanência")
    MonthlyCharges: float = Field(ge=0, description="Valor mensal cobrado")
    TotalCharges: Optional[Union[float, str]] = Field(
        default=None,
        description="Cobrança acumulada (número ou string; vazio vira ausente)",
    )

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def empty_total_charges_as_none(cls, v: Any) -> Any:
        if v is None or v == "":
            return None
        return v

    def to_dataframe(self) -> pd.DataFrame:
        """Converte a linha validada em `DataFrame` 1×p para `predict_proba`."""
        row = self.model_dump()
        return pd.DataFrame([row])


class PredictResponse(BaseModel):
    probability_churn: float = Field(ge=0.0, le=1.0)
    churn_predicted: int = Field(ge=0, le=1)


class HealthResponse(BaseModel):
    status: str
    model_source: str = Field(description="default_synthetic | joblib_file")
