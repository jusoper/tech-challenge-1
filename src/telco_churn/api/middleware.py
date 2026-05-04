"""Middleware HTTP: latência e correlação (Etapa 3 — tarefa 5)."""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

request_logger = logging.getLogger("telco_churn.api.request")


class LatencyRequestMiddleware(BaseHTTPMiddleware):
    """
    Mede tempo de processamento, expõe `X-Process-Time` (segundos, string)
    e `X-Request-ID` (Material APIs — cap. 4: middleware de tempo de resposta).
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_s = time.perf_counter() - start
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{elapsed_s:.6f}"
        request_logger.info(
            "http_request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": round(elapsed_s * 1000, 3),
            },
        )
        return response
