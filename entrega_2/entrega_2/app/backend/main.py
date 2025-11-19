# app/backend/main.py
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(title="SodAI Drinks – Backend")

# ================= Config =================
BASE_DIR = os.getenv("BASE_DIR", "/models")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "rf_pipeline_optuna.joblib")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5").replace(",", "."))

MODEL: Optional[Any] = None
MODEL_PATH: Optional[str] = None

# ================ Utils: modelo =================
def _find_latest_model() -> Optional[str]:
    base = Path(BASE_DIR)
    if not base.exists():
        return None
    for d in sorted([p for p in base.iterdir() if p.is_dir()], reverse=True):
        mdir = d / "models"
        if not mdir.exists():
            continue
        pref = mdir / MODEL_FILENAME
        if pref.exists():
            return str(pref)
        cand = sorted(mdir.glob("*.joblib"), reverse=True)
        if cand:
            return str(cand[0])
    return None

def _load(path: str):
    global MODEL, MODEL_PATH
    MODEL = joblib.load(path)
    MODEL_PATH = path

def _find_ct_and_inputs(model) -> Tuple[Optional[Any], Optional[List[str]]]:
    """
    Busca dentro de un Pipeline el ColumnTransformer (u objeto similar)
    y retorna su lista de columnas de entrada con las que fue fit.
    """
    # Pipeline sklearn tiene atributo steps (lista de (name, step))
    if hasattr(model, "steps"):
        for name, step in model.steps:
            # ColumnTransformer expone 'transformers_' y 'feature_names_in_'
            if hasattr(step, "transformers_") and hasattr(step, "feature_names_in_"):
                return step, list(step.feature_names_in_)
    # Si el propio modelo es el CT
    if hasattr(model, "transformers_") and hasattr(model, "feature_names_in_"):
        return model, list(model.feature_names_in_)
    # Fallback: intentar con feature_names_in_ del objeto raíz
    if hasattr(model, "feature_names_in_"):
        return None, list(model.feature_names_in_)
    return None, None

# ================ Limpieza de payload ================
_NUMERIC_FIELDS = {
    "recency_weeks", "freq_w_12", "size", "num_deliver_per_week", "num_visit_per_week"
}
_CATEGORICAL_FIELDS = {
    "brand", "region_id", "category", "sub_category", "segment", "package", "customer_type",
}

def _to_float_maybe(v):
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(",", ".")
        if s == "" or s.upper() == "DESCONOCIDO":
            return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan
    return np.nan

def _clean_cat(v):
    if v is None:
        return np.nan
    if isinstance(v, str):
        s = v.strip()
        return np.nan if s == "" or s.upper() == "DESCONOCIDO" else s
    # fuerza a string (OneHotEncoder lo tolera con handle_unknown="ignore")
    return str(v)

# ================ Feature engineering (igual que train) ================
def add_deterministic_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # days_since_last_cp
    if "recency_weeks" in X.columns:
        X["recency_weeks"] = pd.to_numeric(X["recency_weeks"], errors="coerce")
        X["days_since_last_cp"] = X["recency_weeks"] * 7.0
    else:
        X["days_since_last_cp"] = 999.0
    X["days_since_last_cp"] = X["days_since_last_cp"].fillna(999.0)

    # desde freq_w_12
    if "freq_w_12" in X.columns:
        base = pd.to_numeric(X["freq_w_12"], errors="coerce").fillna(0.0)
    else:
        base = pd.Series(0.0, index=X.index)
    X["buys_cp_12w"] = base
    X["buys_cp_8w"]  = base * (8.0/12.0)
    X["buys_cp_4w"]  = base * (4.0/12.0)

    # actividad
    nv = pd.to_numeric(X.get("num_visit_per_week", 0.0), errors="coerce").fillna(0.0)
    nd = pd.to_numeric(X.get("num_deliver_per_week", 0.0), errors="coerce").fillna(0.0)
    X["activity_intensity"] = nv + nd

    # recency_inverse + flags
    ds = pd.to_numeric(X["days_since_last_cp"], errors="coerce").fillna(999.0)
    X["recency_inverse"] = 1.0 / (1.0 + ds.replace(0, 0.1))
    X["recent_7d"]  = (ds <= 7).astype(int)
    X["recent_14d"] = (ds <= 14).astype(int)

    # recency_bin (como string ordenable)
    bins = [-1, 7, 14, 28, 56, 84, 999, 10_000]
    labels = ["≤7","≤14","≤28","≤56","≤84","≤999",">999"]
    X["recency_bin"] = pd.cut(ds, bins=bins, labels=labels, ordered=True).astype(str)

    # normalizaciones
    X["freq_12w_norm"] = base
    X["freq_8w_norm"]  = base * (8.0/12.0)

    return X

def _payload_to_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    if "features" in payload and isinstance(payload["features"], dict):
        d = payload["features"]
    elif "records" in payload and isinstance(payload["records"], list) and payload["records"]:
        d = payload["records"][0]
    else:
        raise HTTPException(status_code=400,
            detail="Esperado {'features': {...}} o {'records': [{...}]}")

    # fila base limpia
    row = {}
    for c in _NUMERIC_FIELDS:
        row[c] = _to_float_maybe(d.get(c))
    for c in _CATEGORICAL_FIELDS:
        row[c] = _clean_cat(d.get(c))

    df = pd.DataFrame([row])

    # features derivadas
    df = add_deterministic_features(df)

    # Reindexar según el ColumnTransformer del pipeline
    _, required = _find_ct_and_inputs(MODEL)
    if required:
        for col in required:
            if col not in df.columns:
                # default seguro: num -> 0.0 / cat -> np.nan
                df[col] = 0.0 if col in _NUMERIC_FIELDS or col.startswith(("recent_", "buys_cp_", "freq_", "activity_", "days_since_")) else np.nan
        df = df[required]

    return df

# ================== API ==================
@app.get("/health")
def health():
    return {
        "status": "ready" if MODEL is not None else "missing",
        "base_dir": BASE_DIR,
        "model_date": Path(MODEL_PATH).parts[-3] if MODEL_PATH else (
            sorted([p.name for p in Path(BASE_DIR).glob("*") if p.is_dir()], reverse=True)[0]
            if Path(BASE_DIR).exists() else None
        ),
        "model_path": MODEL_PATH,
        "threshold": THRESHOLD,
        "expected_pattern": "/models/<YYYY-MM-DD>/models/rf_pipeline_optuna.joblib (o rf_pipeline.joblib)",
    }

@app.post("/reload-model")
def reload_model():
    path = _find_latest_model()
    if not path:
        raise HTTPException(status_code=503, detail="No se encontró ningún .joblib en /models/<YYYY-MM-DD>/models/")
    try:
        _load(path)
        return {"status": "ready", "model_path": MODEL_PATH}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar modelo: {type(e).__name__}: {e}")

@app.post("/model/reload")
def reload_model_alias():
    return reload_model()

@app.post("/predict")
def predict(payload: Dict[str, Any]):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado. Llame a /reload-model primero.")

    try:
        df = _payload_to_dataframe(payload)
        proba = float(MODEL.predict_proba(df)[:, 1][0])
        pred  = int(proba >= THRESHOLD)
    except Exception as e:
        # devuélvelo como 400 para ver el motivo en la UI
        raise HTTPException(status_code=400, detail=f"inference_error: {type(e).__name__}: {e}")

    return {
        "probability": proba,
        "prob_next_week": proba,
        "probabilities": [proba],   # UI usa idx 0
        "predictions":   [pred],    # UI usa idx 0
        "will_buy": bool(pred),
        "threshold": THRESHOLD,
        "cols": list(df.columns),
    }

@app.on_event("startup")
def load_model_on_startup():
    try:
        path = _find_latest_model()
        if path:
            _load(path)
            print(f"[startup] Modelo cargado: {path}")
    except Exception as e:
        print(f"[startup] Error al cargar modelo: {e}")
