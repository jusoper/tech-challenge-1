"""Aplicação FastAPI: `/health` e `/predict` (Etapa 3 — tarefa 4)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from sklearn.pipeline import Pipeline

from telco_churn.api.logging_config import configure_api_logging
from telco_churn.api.middleware import LatencyRequestMiddleware
from telco_churn.api.model_runtime import load_or_fit_serving_pipeline
from telco_churn.api.schemas import HealthResponse, PredictResponse, TelcoInferenceRow

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_api_logging()
    model, source = load_or_fit_serving_pipeline()
    app.state.model = model
    app.state.model_source = source
    yield
    app.state.model = None


app = FastAPI(
    title="Telco Churn Inference",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(LatencyRequestMiddleware)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    """Liveness: confirma que o processo está no ar e se o modelo foi carregado."""
    model = getattr(request.app.state, "model", None)
    src = getattr(request.app.state, "model_source", "unknown")
    if model is None:
        raise HTTPException(status_code=503, detail="modelo não inicializado")
    logger.info(
        "health_ok",
        extra={"model_source": src},
    )
    return HealthResponse(status="ok", model_source=src)


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(row: TelcoInferenceRow, request: Request) -> PredictResponse:
    """
    Retorna probabilidade de churn (classe positiva) e rótulo em limiar 0,5.

    O corpo JSON deve incluir pelo menos `tenure`, `MonthlyCharges` e pode incluir
    quaisquer outras colunas aceitas pelo `ColumnTransformer` (ex.: `gender`, `Partner`).
    """
    model: Optional[Pipeline] = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="modelo não inicializado")
    try:
        X = row.to_dataframe()
        proba = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        logger.exception("falha em predict_proba")
        raise HTTPException(status_code=400, detail=f"erro na inferência: {e}") from e
    label = int(proba >= 0.5)
    logger.info(
        "predict_ok",
        extra={
            "churn_predicted": label,
            "probability_churn": round(proba, 6),
        },
    )
    return PredictResponse(probability_churn=proba, churn_predicted=label)
