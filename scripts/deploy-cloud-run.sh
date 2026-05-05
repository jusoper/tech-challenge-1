#!/usr/bin/env bash
# Deploy da API FastAPI no Google Cloud Run.
# Pré-requisitos: Google Cloud SDK (`gcloud`), conta GCP com faturamento ativo,
# autenticação: `gcloud auth login`
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROJECT_ID="${GCP_PROJECT_ID:-tech-challenge-1-495400}"
REGION="${GCP_REGION:-southamerica-east1}"
SERVICE="${CLOUD_RUN_SERVICE:-telco-churn-api}"

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300 \
  --cpu 2

echo "Teste: curl -sS \"\$(gcloud run services describe \"$SERVICE\" --region \"$REGION\" --format='value(status.url)')/health\""
