# Tech Challenge FIAP — Churn (Telco)

## Descrição do projeto

Operadora de telecom perde clientes (churn). O objetivo deste repositório é um pipeline profissional **end-to-end**: dados tabulares públicos (IBM Telco), **baselines** Scikit-learn, **rede neural MLP** em PyTorch (Etapa 2), **MLflow** para experimentos, **API** FastAPI (Etapa 3), testes automatizados e documentação (Model Card na Etapa 4), conforme `etapas-tech-challenge.txt`.

## Etapa 1 (concluída neste branch)

- **ML Canvas:** `docs/ml-canvas.md` (stakeholders, KPIs, SLOs, métricas — alinhado a *Ciclo de Vida*, Aula 01).
- **EDA + baselines + MLflow:** `notebooks/01_eda_baselines.ipynb` (*Ciclo de Vida* Aulas 01–02; *Fundamentos* Aulas 01–02; métricas ROC/PR/F1 em `aula_06_machine-learning-aula-06.txt`).
- **Dados:** script `scripts/download_data.py` (IBM Telco Customer Churn, CSV público).

### Setup

Requer **Python 3.9+** (recomendado 3.11+).

```bash
cd tech-challenge-1
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"   # inclui ruff; pytest já vem na instalação base
python scripts/download_data.py
```

### Makefile (Etapa 3 — lint, teste, API)

Na raiz do repositório (`tech-challenge-1/`). O `Makefile` usa `.venv/bin/python` se existir; caso contrário, `python3`.

| Comando | Descrição |
|--------|------------|
| `make dev` | `pip install -e ".[dev]"` (instala **ruff** para `lint`) |
| `make lint` | `ruff check src tests scripts` |
| `make test` | `pytest` |
| `make run` | sobe a API com **uvicorn** em `http://0.0.0.0:8000` |
| `make check` | `lint` + `test` (útil em CI ou antes de commit) |

### Notebook e MLflow

```bash
jupyter notebook notebooks/01_eda_baselines.ipynb
```

Os runs são gravados em `mlruns/` (tracking local). Para abrir a UI:

```bash
mlflow ui --backend-store-uri file:$(pwd)/mlruns
```

## Estrutura

| Pasta | Uso |
|--------|-----|
| `src/telco_churn/` | Código do pacote (Etapas 2–3) |
| `data/raw/` | CSV baixado (não versionado) |
| `notebooks/` | EDA e experimentos |
| `docs/` | ML Canvas, Model Card, [deploy](docs/deploy-architecture.md), [monitoramento](docs/monitoring-plan.md) (Etapa 4) |
| `models/` | Artefatos salvos |
| `tests/` | Pytest |

## Próximas etapas

- **Etapa 4:** Model Card, README final, vídeo STAR, deploy opcional.
