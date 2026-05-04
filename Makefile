# Etapa 3 — tarefa 6: alvos padronizados (Eng. Software, Aulas 04–05)
# Uso: com venv ativo ou com .venv/ criado em `tech-challenge-1/`.

PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: help install dev lint test run check

help:
	@echo "Targets:"
	@echo "  make install   pip install -e ."
	@echo "  make dev       pip install -e \".[dev]\"  (ruff para lint)"
	@echo "  make lint      ruff check src tests scripts"
	@echo "  make test      pytest"
	@echo "  make run       API FastAPI (uvicorn :8000)"
	@echo "  make check     lint + test"

install:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check src tests scripts

test:
	$(PYTHON) -m pytest tests/

run:
	$(PYTHON) -m uvicorn telco_churn.api.main:app --host 0.0.0.0 --port 8000

check: lint test
