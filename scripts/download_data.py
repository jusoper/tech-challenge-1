"""Baixa o dataset IBM Telco Customer Churn (referência pública do tech challenge)."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlretrieve

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "raw" / "Telco-Customer-Churn.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"Arquivo já existe: {out}")
        return
    print(f"Baixando para {out} ...")
    urlretrieve(URL, out)
    print("Concluído.")


if __name__ == "__main__":
    main()
    sys.exit(0)
