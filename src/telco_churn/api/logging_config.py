"""Logging estruturado em JSON para a API (Etapa 3 — tarefa 5)."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

_RECORD_INTERNAL = frozenset(
    logging.LogRecord(
        name="",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    ).__dict__.keys()
) | {"message", "asctime"}


class JsonLogFormatter(logging.Formatter):
    """Uma linha JSON por evento (consumo por agregadores / Loki / CloudWatch)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RECORD_INTERNAL or key.startswith("_"):
                continue
            if value is not None:
                try:
                    json.dumps(value)
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = repr(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_api_logging(*, level: str | None = None) -> None:
    """
    Configura `StreamHandler` JSON no logger `telco_churn` (idempotente em testes).
    Nível: env `TELCO_LOG_LEVEL` ou `level` ou INFO.
    """
    lvl = (level or os.environ.get("TELCO_LOG_LEVEL", "INFO")).upper()
    log = logging.getLogger("telco_churn")
    log.setLevel(getattr(logging, lvl, logging.INFO))
    has_json = any(
        isinstance(h.formatter, JsonLogFormatter)
        for h in log.handlers
        if getattr(h, "formatter", None) is not None
    )
    if not has_json:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonLogFormatter())
        log.addHandler(handler)
    log.propagate = False
