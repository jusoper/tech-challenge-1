"""Smoke mínimo: garante layout esperado pelo roteiro (Etapas 3–4 expandem testes)."""

from pathlib import Path


def test_pyproject_and_notebook_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "pyproject.toml").is_file()
    assert (root / "notebooks" / "01_eda_baselines.ipynb").is_file()


def test_download_script_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "download_data.py").is_file()


def test_telco_churn_domain_subpackages():
    """Etapa 3 — tarefa 1: módulos por responsabilidade (Eng. Software, Aula 01)."""
    root = Path(__file__).resolve().parents[1] / "src" / "telco_churn"
    for sub in ("data", "modeling", "training", "evaluation", "business", "tracking"):
        assert (root / sub / "__init__.py").is_file()
