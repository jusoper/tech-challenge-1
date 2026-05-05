FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip && pip install .

# Inclua artefatos versionados em models/ quando existirem; descomente se necessário:
# COPY models ./models

EXPOSE 8080

CMD exec uvicorn telco_churn.api.main:app --host 0.0.0.0 --port ${PORT:-8080}
