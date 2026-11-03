"""Synthetic tabular dataset with a REAL learnable signal, for rehearsing the
tabular pipeline end to end tonight.

    python ml/make_sample_csv.py
    curl -X POST localhost:8000/ai/tabular/train -H 'Content-Type: application/json' \
         -d '{"csv_path":"data/raw/sample.csv","target":"resolved_late"}'
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "sample.csv"


def main(n: int = 3000) -> None:
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "severity": rng.integers(1, 6, n),
        "days_open": rng.gamma(2.0, 4.0, n).round(1),
        "department": rng.choice(["health", "pwd", "education", "police", "sanitation"], n),
        "channel": rng.choice(["web", "whatsapp", "ivr", "field"], n, p=[0.3, 0.35, 0.2, 0.15]),
        "ward_population": rng.integers(2000, 60000, n),
        "staff_on_duty": rng.integers(1, 12, n),
        "prior_complaints": rng.poisson(2.2, n),
        "monsoon": rng.choice([0, 1], n, p=[0.7, 0.3]),
    })

    # Deliberate, explainable signal: understaffing + severity + backlog drive lateness.
    logit = (
        0.42 * df.severity
        + 0.06 * df.days_open
        - 0.19 * df.staff_on_duty
        + 0.13 * df.prior_complaints
        + 0.55 * df.monsoon
        + df.department.map({"health": 0.1, "pwd": 0.5, "education": -0.2, "police": 0.0, "sanitation": 0.35})
        - 2.1
    )
    p = 1 / (1 + np.exp(-logit))
    df["resolved_late"] = rng.binomial(1, p)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT}  rows={len(df)}  positive_rate={df.resolved_late.mean():.3f}")
    print("Target column: resolved_late")


if __name__ == "__main__":
    main()
