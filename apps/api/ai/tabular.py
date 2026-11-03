"""Tabular prediction: train on a CSV, predict, explain via feature importance.

Covers the huge class of PS statements that are really 'predict a number or a
class from a spreadsheet': risk scoring, demand forecasting, triage, fraud,
yield, churn, dropout, no-show.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from core.config import ROOT, settings

_MODEL = settings.MODEL_DIR / "tabular.pkl"
_META = settings.MODEL_DIR / "tabular_meta.json"


def resolve(csv_path: str) -> Path:
    """Accept an absolute path, or one relative to the repo root.

    The API process runs from apps/api, so a bare 'data/raw/x.csv' would
    otherwise resolve to the wrong place. This is a 20-minute bug at 3am.
    """
    p = Path(csv_path).expanduser()
    for cand in (p, ROOT / p, Path.cwd() / p):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"CSV not found: {csv_path}. Tried: {p}, {ROOT / p}. "
        f"Use a path relative to the repo root, e.g. data/raw/sample.csv"
    )


def train(csv_path: str, target: str, task: str = "auto", test_size: float = 0.2) -> dict[str, Any]:
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = pd.read_csv(resolve(csv_path))
    if target not in df.columns:
        raise ValueError(f"target '{target}' not in columns: {list(df.columns)}")
    df = df.dropna(subset=[target])
    y = df[target]
    X = df.drop(columns=[target])

    if task == "auto":
        task = "classification" if (y.dtype == object or y.nunique() <= 20) else "regression"

    num = [c for c in X.columns if X[c].dtype.kind in "ifb"]
    cat = [c for c in X.columns if c not in num]

    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", max_categories=25))]), cat),
        ]
    )
    model = GradientBoostingClassifier(random_state=0) if task == "classification" else GradientBoostingRegressor(random_state=0)
    pipe = Pipeline([("pre", pre), ("model", model)])

    strat = y if (task == "classification" and y.value_counts().min() >= 2) else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=0, stratify=strat)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    metrics = (
        {"accuracy": round(float(accuracy_score(yte, pred)), 4), "f1_weighted": round(float(f1_score(yte, pred, average="weighted")), 4)}
        if task == "classification"
        else {"r2": round(float(r2_score(yte, pred)), 4), "mae": round(float(mean_absolute_error(yte, pred)), 4)}
    )

    try:
        names = list(pipe.named_steps["pre"].get_feature_names_out())
        imps = pipe.named_steps["model"].feature_importances_
        top = sorted(zip(names, imps.tolist()), key=lambda t: -t[1])[:12]
        importance = [{"feature": n.split("__")[-1], "weight": round(float(w), 4)} for n, w in top]
    except Exception:
        importance = []

    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(_MODEL, "wb") as f:
        pickle.dump(pipe, f)
    meta = {
        "task": task, "target": target, "features": list(X.columns),
        "numeric": num, "categorical": cat, "metrics": metrics,
        "importance": importance, "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "classes": sorted(map(str, y.unique().tolist())) if task == "classification" else None,
    }
    _META.write_text(json.dumps(meta, indent=2))
    return meta


def predict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import pandas as pd

    if not _MODEL.exists():
        return {"ok": False, "error": "No tabular model trained. POST /ai/tabular/train first."}
    meta = json.loads(_META.read_text()) if _META.exists() else {}
    with open(_MODEL, "rb") as f:
        pipe = pickle.load(f)
    df = pd.DataFrame(rows)
    for c in meta.get("features", []):
        if c not in df.columns:
            df[c] = None
    df = df[meta.get("features", list(df.columns))]

    preds = pipe.predict(df).tolist()
    out: dict[str, Any] = {"ok": True, "task": meta.get("task"), "predictions": preds}
    if meta.get("task") == "classification" and hasattr(pipe, "predict_proba"):
        try:
            out["probabilities"] = [
                {str(c): round(float(p), 4) for c, p in zip(pipe.classes_, row)} for row in pipe.predict_proba(df)
            ]
        except Exception:
            pass
    out["importance"] = meta.get("importance", [])
    return out


def available() -> dict[str, Any]:
    meta = json.loads(_META.read_text()) if _META.exists() else {}
    return {"ok": True, "trained": _MODEL.exists(), "task": meta.get("task"), "metrics": meta.get("metrics", {}), "target": meta.get("target")}
