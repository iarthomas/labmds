from pathlib import Path
import pandas as pd
import numpy as np

# ----------------------------- #
# 1) Crear carpetas (data/YYYY-MM-DD/{raw,splits,models})
# ----------------------------- #

def create_folders(base_dir: str = "data", **kwargs) -> str:
    """
    Crea una carpeta con nombre de la fecha de ejecución (YYYY-MM-DD) y,
    dentro, las subcarpetas: raw, splits, models.
    Retorna la ruta creada como string.
    """
    date_str = (
        kwargs.get("ds")
        or (kwargs.get("logical_date").strftime("%Y-%m-%d") if kwargs.get("logical_date") else None)
        or (kwargs.get("execution_date").strftime("%Y-%m-%d") if kwargs.get("execution_date") else None)
    )
    if not date_str:
        raise ValueError("No se encontró la fecha de ejecución en el contexto (ds/logical_date/execution_date).")

    root = Path(base_dir) / date_str
    for sub in ("raw", "splits", "models"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    return str(root)


# ----------------------------- #
# 2) Código para construir el dataset supervisado
# ----------------------------- #

# Helpers
def _safe_str(s: pd.Series) -> pd.Series:
    """Convierte a dtype string, recorta espacios y estandariza vacíos como <NA>."""
    out = s.astype("string")
    out = out.str.strip()
    out = out.replace(r"^\s*$", pd.NA, regex=True)
    return out

def _ensure_datetime_localized(s: pd.Series, tz: str = "UTC") -> pd.Series:
    """
    Parsea a datetime y asegura zona horaria.
    - Si las fechas son naïve: se localizan en 'tz' (o UTC).
    - Si ya tienen tz: se convierten a 'tz' si es distinto.
    Devuelve timestamps con tz (por defecto UTC).
    """
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    try:
        # Si es naïve, localizar
        if dt.dt.tz is None:
            if tz and tz.upper() != "UTC":
                dt = dt.dt.tz_localize(tz, nonexistent="NaT", ambiguous="NaT")
            else:
                dt = dt.dt.tz_localize("UTC")
        # Convertir si se pidió otra zona
        if tz and tz.upper() != "UTC":
            dt = dt.dt.tz_convert(tz)
    except Exception:
        # Fallback conservador si algo falla
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        if tz and tz.upper() != "UTC":
            try:
                dt = dt.dt.tz_convert(tz)
            except Exception:
                pass
    return dt

def _week_start(s: pd.Series) -> pd.Series:
    """Inicio de semana (lunes 00:00) preservando TZ."""
    norm = s.dt.normalize()
    return norm - pd.to_timedelta(norm.dt.weekday, unit="D")

# Limpiezas por tabla
def clean_clientes_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["customer_id"]   = _safe_str(df["customer_id"])
    df["region_id"]     = _safe_str(df["region_id"])
    df["customer_type"] = _safe_str(df.get("customer_type", pd.Series(index=df.index))).fillna("DESCONOCIDO")

    for col in ["Y", "X"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["num_deliver_per_week", "num_visit_per_week"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("Int64")
        else:
            df[col] = pd.Series(0, index=df.index, dtype="Int64")

    before = len(df)
    # Deduplicar por customer_id 
    df = df.drop_duplicates(subset=["customer_id"], keep="last")
    df = df.dropna(subset=["customer_id", "region_id"])
    print(f"[clientes(df)] cargados={before}, únicos={len(df)}")
    return df.reset_index(drop=True)

def clean_productos_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["product_id"] = _safe_str(df["product_id"])
    for col in ["brand", "category", "sub_category", "segment", "package"]:
        if col in df.columns:
            df[col] = _safe_str(df[col]).fillna("DESCONOCIDO")
        else:
            df[col] = "DESCONOCIDO"

    if "size" in df.columns:
        df["size"] = pd.to_numeric(df["size"], errors="coerce")
    else:
        df["size"] = np.nan

    before = len(df)
    df = df.drop_duplicates(subset=["product_id"], keep="last")
    print(f"[productos(df)] cargados={before}, únicos={len(df)}")
    return df.reset_index(drop=True)

def clean_transacciones_df(df: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    df = df.copy()
    df["customer_id"] = _safe_str(df["customer_id"])
    df["product_id"]  = _safe_str(df["product_id"])

    df["purchase_date"] = _ensure_datetime_localized(df["purchase_date"], tz=tz)
    if "payment" in df.columns:
        df["payment"] = pd.to_numeric(df["payment"], errors="coerce")
    else:
        df["payment"] = np.nan

    df = df.dropna(subset=["customer_id", "product_id", "purchase_date"])

    subset = [c for c in ["order_id", "customer_id", "product_id", "purchase_date"] if c in df.columns]
    before = len(df)
    if subset:
        df = df.sort_values(["purchase_date"]).drop_duplicates(subset=subset, keep="first")

    df["week_start"] = _week_start(df["purchase_date"])
    print(f"[transacciones(df)] cargadas={before}, únicas={len(df)}, semanas={df['week_start'].nunique()}")
    return df.reset_index(drop=True)

# Armado de dataset supervisado
def build_supervised_dataset(
    clientes: pd.DataFrame,
    productos: pd.DataFrame,
    tx: pd.DataFrame,
    lookback_weeks: int = 12,
    top_region_type: int = 20,
) -> pd.DataFrame:
    # 1) Subset de clientes por regiones más frecuentes
    if "region_id" in clientes.columns:
        top_regs = (clientes["region_id"].value_counts().head(top_region_type).index)
        clientes_f = clientes[clientes["region_id"].isin(top_regs)].copy()
        # Reducir transacciones a esos clientes para mejorar performance
        tx = tx[tx["customer_id"].isin(clientes_f["customer_id"])].copy()
    else:
        clientes_f = clientes.copy()

    # 2) Agregación semanal por cliente–producto
    if tx.empty:
        # Si no hay transacciones, devolver dataframe vacío con columnas esperadas mínimas
        cols = [
            "week_start", "customer_id", "product_id", "y_next_week",
            "has_purchase", "n_purchases", "pay_sum",
            f"freq_w_{lookback_weeks}", f"pay_sum_{lookback_weeks}",
            "week_idx", "recency_weeks",
            "region_id", "customer_type", "num_deliver_per_week", "num_visit_per_week", "X", "Y",
            "brand", "category", "sub_category", "segment", "package", "size",
        ]
        return pd.DataFrame(columns=[c for c in cols if c in (list(clientes_f.columns) + list(productos.columns) + cols)])

    agg = (
        tx.groupby(["customer_id", "product_id", "week_start"])
          .agg(
              n_purchases=("order_id", "nunique") if "order_id" in tx.columns else ("product_id", "size"),
              pay_sum=("payment", "sum")
          )
          .reset_index()
    )
    agg["has_purchase"] = (agg["n_purchases"] > 0).astype(int)
    agg["pay_sum"] = agg["pay_sum"].fillna(0.0)

    # 3) Calendario semanal completo por par (customer_id, product_id)
    wk_min = agg["week_start"].min()
    wk_max = agg["week_start"].max()
    tz = getattr(agg["week_start"].dt, "tz", None)
    all_weeks = pd.date_range(wk_min, wk_max, freq="W-MON", tz=tz)

    pairs = agg[["customer_id", "product_id"]].drop_duplicates()
    pairs = pairs.assign(_k=1)
    weeks = pd.DataFrame({"week_start": all_weeks, "_k": 1})
    grid = pairs.merge(weeks, on="_k", how="outer").drop(columns="_k")

    # Mezcla con compras observadas; faltantes = 0
    dfw = (
        grid.merge(
            agg[["customer_id", "product_id", "week_start", "has_purchase", "n_purchases", "pay_sum"]],
            on=["customer_id", "product_id", "week_start"],
            how="left",
        )
        .sort_values(["customer_id", "product_id", "week_start"])
    )
    dfw[["has_purchase", "n_purchases", "pay_sum"]] = dfw[["has_purchase", "n_purchases", "pay_sum"]].fillna(
        {"has_purchase": 0, "n_purchases": 0, "pay_sum": 0.0}
    )

    # 4) Features de historial (rolling excluyendo la semana actual)
    def _add_roll_feats(g):
        g = g.sort_values("week_start").copy()
        g[f"freq_w_{lookback_weeks}"] = (
            g["has_purchase"].rolling(window=lookback_weeks, min_periods=1).sum().shift(1)
        )
        g[f"pay_sum_{lookback_weeks}"] = (
            g["pay_sum"].rolling(window=lookback_weeks, min_periods=1).sum().shift(1)
        )
        g["week_idx"] = np.arange(len(g))
        last_buy_idx = g["week_idx"].where(g["has_purchase"] == 1).ffill()
        g["recency_weeks"] = (g["week_idx"] - last_buy_idx).fillna(999).astype(int)
        g["y_next_week"] = g["has_purchase"].shift(-1).fillna(0).astype(int)
        return g

    dfw = dfw.groupby(["customer_id", "product_id"], group_keys=False).apply(_add_roll_feats)

    # 5) Merge con metadata de cliente y producto (features categóricas)
    keep_cli = [c for c in ["customer_id", "region_id", "customer_type", "num_deliver_per_week",
                            "num_visit_per_week", "X", "Y"] if c in clientes_f.columns]
    keep_prd = [c for c in ["product_id", "brand", "category", "sub_category", "segment", "package", "size"]
                if c in productos.columns]
    cli_min = clientes_f[keep_cli].drop_duplicates("customer_id")
    prd_min = productos[keep_prd].drop_duplicates("product_id")

    ds = (
        dfw.merge(cli_min, on="customer_id", how="inner")
           .merge(prd_min, on="product_id", how="left")
    )

    # 6) Limpiezas finales para modelar y orden de columnas
    ds = ds.dropna(subset=["y_next_week"])

    desired_order = [
        "week_start", "customer_id", "product_id", "y_next_week",
        "has_purchase", "n_purchases", "pay_sum",
        f"freq_w_{lookback_weeks}", f"pay_sum_{lookback_weeks}",
        "week_idx", "recency_weeks",
        "region_id", "customer_type", "num_deliver_per_week", "num_visit_per_week", "X", "Y",
        "brand", "category", "sub_category", "segment", "package", "size",
    ]
    ordered = [c for c in desired_order if c in ds.columns] + [c for c in ds.columns if c not in desired_order]
    ds = ds[ordered].reset_index(drop=True)

    return ds

# Función de entrada (aplica todo)
def build_from_dataframes(
    df_clientes: pd.DataFrame,
    df_productos: pd.DataFrame,
    df_transacciones: pd.DataFrame,
    lookback_weeks: int = 12,
    top_region_type: int = 20,
    tz: str = "UTC",
) -> pd.DataFrame:
    """
    Aplica limpieza a las tres tablas y construye el dataset supervisado.
    """
    clientes  = clean_clientes_df(df_clientes)
    productos = clean_productos_df(df_productos)
    tx        = clean_transacciones_df(df_transacciones, tz=tz)

    ds = build_supervised_dataset(
        clientes=clientes,
        productos=productos,
        tx=tx,
        lookback_weeks=lookback_weeks,
        top_region_type=top_region_type,
    )
    return ds


# ----------------------------- #
# 3) Guardar el dataset supervisado en la carpeta raw
# ----------------------------- #

def save_supervised_to_raw(
    dataset: pd.DataFrame,
    base_dir: str = "data",
    filename: str = "supervised.parquet",
    **kwargs,
) -> str:
    """
    Crea (si no existen) las carpetas data/YYYY-MM-DD/{raw,splits,models}
    y guarda `ds` en data/YYYY-MM-DD/raw/{filename}.
    Retorna la ruta del archivo guardado.
    """
    root = create_folders(base_dir=base_dir, **kwargs)
    out_path = Path(root) / "raw" / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        dataset.to_csv(out_path, index=False)
    else:
        dataset.to_parquet(out_path, index=False)
    print(f"[save] supervised -> {out_path}")
    return str(out_path)






from pathlib import Path
import time
import numpy as np
import pandas as pd
import joblib
from typing import List, Dict, Any, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier


# ----------------------------- #
# 4) Hold-out temporal (70/15/15 por semanas) desde raw/supervised.*
# ----------------------------- #

def _temporal_split_by_weeks(
    data: pd.DataFrame,
    date_col: str = "week_start",
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    gap_weeks: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Split temporal por semanas únicas en `date_col`:
      - train = primeras `train_ratio`
      - valid = siguientes `valid_ratio` (con gap opcional)
      - test  = resto (con gap opcional)
    """
    weeks = np.array(sorted(data[date_col].dropna().unique()))
    n = len(weeks)
    assert n >= 10, f"Hay muy pocas semanas ({n}) para un split 70/15/15."

    i_tr_end = int(np.floor(n * train_ratio)) - 1
    i_va_end = int(np.floor(n * (train_ratio + valid_ratio))) - 1
    i_tr_end = max(i_tr_end, 0)
    i_va_end = min(max(i_va_end, i_tr_end + 1), n - 1)

    tr_end_cut  = weeks[i_tr_end]
    va_start_cut = weeks[i_tr_end + 1] if i_tr_end + 1 < n else weeks[i_tr_end]
    va_end_cut   = weeks[i_va_end]
    te_start_cut = weeks[i_va_end + 1] if i_va_end + 1 < n else weeks[i_va_end]

    def add_weeks(dt, k): return dt + pd.Timedelta(days=7 * k)
    tr_max = add_weeks(tr_end_cut, 0)
    va_min = add_weeks(va_start_cut, +gap_weeks)
    va_max = va_end_cut
    te_min = add_weeks(te_start_cut, +gap_weeks)

    train = data[(data[date_col] <= tr_max)]
    valid = data[(data[date_col] >= va_min) & (data[date_col] <= va_max)]
    test  = data[(data[date_col] >= te_min)]

    # Asegurar no solape
    valid = valid[valid[date_col] > tr_max]
    test  = test[test[date_col] > va_max]

    assert train[date_col].max() < valid[date_col].min(), "Train y Valid se solapan (revisar gap/ratios)."
    assert valid[date_col].max() < test[date_col].min(),  "Valid y Test se solapan (revisar gap/ratios)."

    meta = {
        "weeks_total": n,
        "train_weeks": train[date_col].nunique(),
        "valid_weeks": valid[date_col].nunique(),
        "test_weeks":  test[date_col].nunique(),
        "train_rows": len(train),
        "valid_rows": len(valid),
        "test_rows":  len(test),
        "train_start": train[date_col].min(),
        "train_end":   train[date_col].max(),
        "valid_start": valid[date_col].min(),
        "valid_end":   valid[date_col].max(),
        "test_start":  test[date_col].min(),
        "test_end":    test[date_col].max(),
        "gap_weeks":   gap_weeks,
    }
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True), meta


def split_supervised_temporal_holdout(
    base_dir: str = "data",
    supervised_filename: str = "supervised.parquet",
    date_col: str = "week_start",
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    gap_weeks: int = 0,
    **kwargs,
) -> tuple[str, str, str, dict]:
    """
    Lee el dataset supervisado en raw/{supervised_filename} dentro de la carpeta de fecha (YYYY-MM-DD),
    aplica un split temporal por semanas (train/valid/test) con posible `gap_weeks`
    y guarda train.*, valid.*, test.* en splits/.
    Retorna (train_path, valid_path, test_path, meta).
    """
    date_str = (
        kwargs.get("ds")
        or (kwargs.get("logical_date").strftime("%Y-%m-%d") if kwargs.get("logical_date") else None)
        or (kwargs.get("execution_date").strftime("%Y-%m-%d") if kwargs.get("execution_date") else None)
    )
    if not date_str:
        raise ValueError("No se encontró la fecha de ejecución en el contexto (ds/logical_date/execution_date).")

    root = Path(base_dir) / date_str
    raw_path = root / "raw" / supervised_filename
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {raw_path}")

    # Cargar según extensión
    if raw_path.suffix.lower() == ".csv":
        df = pd.read_csv(raw_path)
        out_ext = ".csv"
    elif raw_path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(raw_path)
        out_ext = ".parquet"
    else:
        raise ValueError(f"Extensión no soportada: {raw_path.suffix}. Use .csv o .parquet")

    if date_col not in df.columns:
        raise ValueError(
            f"La columna de fecha '{date_col}' no existe en {raw_path.name}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    # Normalizar fecha sin tocar TZ (para no desalinear semanas)
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Split temporal
    ds_train, ds_valid, ds_test, meta = _temporal_split_by_weeks(
        data=df,
        date_col=date_col,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        gap_weeks=gap_weeks,
    )

    # Guardar
    train_path = splits_dir / f"train{out_ext}"
    valid_path = splits_dir / f"valid{out_ext}"
    test_path  = splits_dir / f"test{out_ext}"

    if out_ext == ".csv":
        ds_train.to_csv(train_path, index=False)
        ds_valid.to_csv(valid_path, index=False)
        ds_test.to_csv(test_path, index=False)
    else:
        ds_train.to_parquet(train_path, index=False)
        ds_valid.to_parquet(valid_path, index=False)
        ds_test.to_parquet(test_path, index=False)

    print(f"[split temporal] train -> {train_path} | valid -> {valid_path} | test -> {test_path}")
    return str(train_path), str(valid_path), str(test_path), meta


# =========================
#     Transformers custom
# =========================

class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip por cuantiles o por límites fijos. Devuelve DataFrame con mismas columnas."""
    def __init__(self, cols: List[str], lower: Optional[float]=None, upper: Optional[float]=None, q_lower=0.0, q_upper=1.0):
        self.cols, self.lower, self.upper, self.q_lower, self.q_upper = cols, lower, upper, q_lower, q_upper
    def fit(self, X, y=None):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.cols)
        self.lims_ = {}
        for c in self.cols:
            if self.lower is not None or self.upper is not None:
                lo = -np.inf if self.lower is None else self.lower
                hi =  np.inf if self.upper is None else self.upper
            else:
                lo = X[c].quantile(self.q_lower)
                hi = X[c].quantile(self.q_upper)
            self.lims_[c] = (lo, hi)
        return self
    def transform(self, X):
        X = X.copy()
        for c,(lo,hi) in self.lims_.items():
            if c in X.columns:
                X[c] = np.clip(X[c], lo, hi)
        return X

class FrequencyEncoderAdd(BaseEstimator, TransformerMixin):
    """Agrega columnas *_freq con la frecuencia relativa de cada categoría (calculada en fit)."""
    def __init__(self, cols: List[str]):
        self.cols = cols
    def fit(self, X, y=None):
        self.maps_ = {}
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.cols)
        for c in self.cols:
            if c in df.columns:
                vc = df[c].astype("string").value_counts(normalize=True)
                self.maps_[c] = vc
            else:
                self.maps_[c] = pd.Series(dtype=float)
        return self
    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            if c in X.columns:
                X[f"{c}_freq"] = X[c].astype("string").map(self.maps_[c]).fillna(0.0).astype(float)
            else:
                X[f"{c}_freq"] = 0.0
        return X

def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# =========================
#  B) Features determinísticas ANTES del pipeline
# =========================

def add_deterministic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features sin "aprender" del conjunto:
      - days_since_last_cp (desde recency_weeks)
      - buys_cp_12w/8w/4w (desde alguna freq_w_*; si no existe, rellena 0)
      - activity_intensity, recency_inverse, recent_7d/recent_14d
      - recency_bin (cortes fijos)
    """
    X = df.copy()

    # Base recency (días)
    if "recency_weeks" in X.columns:
        rec_days = X["recency_weeks"].astype(float) * 7.0
    else:
        rec_days = pd.Series(999.0, index=X.index)
    X["days_since_last_cp"] = rec_days.fillna(999.0)

    # Elegir una freq_w_* para aproximar compras en ventanas
    freq_cols = [c for c in X.columns if c.startswith("freq_w_")]
    if freq_cols:
        # Preferir 12; si no, tomar la ventana más cercana (parsear entero)
        def _win(c):
            try:
                return int(c.split("_")[-1])
            except Exception:
                return None
        if "freq_w_12" in freq_cols:
            chosen = "freq_w_12"
            w = 12
        else:
            # elegir la que tenga ventana válida; si hay varias, la más grande
            valid = [(c, _win(c)) for c in freq_cols if _win(c)]
            if valid:
                chosen, w = sorted(valid, key=lambda t: t[1])[-1]
            else:
                chosen, w = freq_cols[0], 12  # fallback
        base = X[chosen].astype(float).fillna(0.0)
        # Escalar linealmente si la ventana elegida no es 12
        X["buys_cp_12w"] = base * (12.0 / float(w))
        X["buys_cp_8w"]  = base * (8.0  / float(w))
        X["buys_cp_4w"]  = base * (4.0  / float(w))
    else:
        X["buys_cp_12w"] = 0.0
        X["buys_cp_8w"]  = 0.0
        X["buys_cp_4w"]  = 0.0

    # Otros derivados determinísticos
    X["activity_intensity"] = X.get("num_visit_per_week", 0).fillna(0) + X.get("num_deliver_per_week", 0).fillna(0)
    X["recency_inverse"] = 1.0 / (1.0 + X["days_since_last_cp"].replace(0, 0.1).astype(float))
    X["recent_7d"]  = (X["days_since_last_cp"].astype(float) <= 7).astype(int)
    X["recent_14d"] = (X["days_since_last_cp"].astype(float) <= 14).astype(int)

    # Bins fijos
    bins = [-1, 7, 14, 28, 56, 84, 999, 10_000]
    labels = ["≤7","≤14","≤28","≤56","≤84","≤999",">999"]
    X["recency_bin"] = pd.cut(
        X["days_since_last_cp"].astype(float).fillna(999),
        bins=bins, labels=labels, ordered=True
    )
    return X


# =========================
#     Preprocesamiento
# =========================

def build_preprocessor(config: Dict[str, Any]) -> Pipeline:
    """
    config:
      - scale: 'standard' | 'robust' | 'none'
      - make_bins: bool (si True, espera 'recency_bin' ya creada)
      - clip_size_upper: float
      - cat_high: list  (frequency encoding)
      - cat_low: list   (one-hot)
      - num_base: list
      - num_extra: list (ya creadas por add_deterministic_features)
    """
    scaler = {
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "none": "passthrough"
    }[config.get("scale", "robust")]

    clipper  = OutlierClipper(cols=["size"], lower=0.0, upper=config.get("clip_size_upper", 10.0))
    freq_add = FrequencyEncoderAdd(cols=config["cat_high"])

    numeric_cols = (
        config["num_base"]
        + config.get("num_extra", [])
        + [f"{c}_freq" for c in config["cat_high"]]
    )
    cat_ohe_cols = config["cat_low"] + (["recency_bin"] if config.get("make_bins", True) else [])

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", _make_ohe()),
    ])

    coltx = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat_low_ohe", cat_pipe, cat_ohe_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pre = Pipeline([
        ("clip", clipper),
        ("freq", freq_add),
        ("coltx", coltx),
    ])
    return pre


# Config por defecto (ya considerando features creadas antes)
DEFAULT_CONFIG = {
    "scale": "robust",
    "make_bins": True,
    "clip_size_upper": 10.0,
    "cat_high": ["brand", "region_id"],
    "cat_low":  ["category","sub_category","segment","package","customer_type"],
    "num_base": ["days_since_last_cp","buys_cp_4w","buys_cp_8w","buys_cp_12w",
                 "num_deliver_per_week","num_visit_per_week","size"],
    "num_extra": ["activity_intensity","freq_12w_norm","freq_8w_norm","recency_inverse","recent_7d","recent_14d"],
}


def preprocess_and_train(
    base_dir: str = "data",
    train_filename: str = "train.parquet",
    test_filename: str = "test.parquet",
    target_col: str = "y_next_week",
    model_filename: str = "rf_pipeline.joblib",
    config: Dict[str, Any] = None,
    random_state: int = 42,
    **kwargs,
) -> str:
    """
    Lee train/test desde data/YYYY-MM-DD/splits/,
    crea features determinísticas ANTES del pipeline,
    arma Pipeline: (OutlierClipper -> FrequencyEncoderAdd -> ColumnTransformer(OHE+escala)) + RandomForest,
    entrena en TRAIN, evalúa en TEST (accuracy y F1 de la clase positiva),
    y guarda el pipeline entrenado en data/YYYY-MM-DD/models/{model_filename}.
    Retorna la ruta del .joblib guardado.
    """
    cfg = (config or DEFAULT_CONFIG).copy()

    # Resolver fecha
    date_str = (
        kwargs.get("ds")
        or (kwargs.get("logical_date").strftime("%Y-%m-%d") if kwargs.get("logical_date") else None)
        or (kwargs.get("execution_date").strftime("%Y-%m-%d") if kwargs.get("execution_date") else None)
    )
    if not date_str:
        raise ValueError("No se encontró la fecha de ejecución en el contexto (ds/logical_date/execution_date).")

    root = Path(base_dir) / date_str
    splits_dir = root / "splits"
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / train_filename
    test_path  = splits_dir / test_filename
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"No se encontró train/test en {splits_dir}. Faltan: "
                                f"{'[OK]' if train_path.exists() else train_path} | "
                                f"{'[OK]' if test_path.exists() else test_path}")

    # Cargar según extensión
    def _load_any(p: Path) -> pd.DataFrame:
        suf = p.suffix.lower()
        if suf == ".csv":
            return pd.read_csv(p)
        if suf in (".parquet", ".pq"):
            return pd.read_parquet(p)
        raise ValueError(f"Extensión no soportada: {suf}. Use .csv o .parquet")

    train_df = _load_any(train_path)
    test_df  = _load_any(test_path)

    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise ValueError(f"Columna objetivo '{target_col}' no está en train/test.")

    # Quitar llaves y target
    drop_cols = [c for c in ["week_start", "customer_id", "product_id"] if c in train_df.columns]
    X_train = train_df.drop(columns=[target_col] + drop_cols)
    y_train = train_df[target_col].astype(int).values
    X_test  = test_df.drop(columns=[target_col] + drop_cols)
    y_test  = test_df[target_col].astype(int).values

    # Validaciones mínimas de "materia prima"
    if "recency_weeks" not in X_train.columns:
        raise KeyError("Falta 'recency_weeks' en el dataset para derivar 'days_since_last_cp'.")
    if not any(c.startswith("freq_w_") for c in X_train.columns):
        raise KeyError("Se necesita al menos una columna 'freq_w_*' (p. ej. 'freq_w_12') para derivar 'buys_cp_*'.")

    # Crear features determinísticas ANTES del pipeline
    X_train = add_deterministic_features(X_train)
    X_test  = add_deterministic_features(X_test)

    # Normalizaciones extra (derivadas simples)
    # Nota: si quieres usar estas, agrega 'freq_12w_norm' y 'freq_8w_norm' en num_extra del config
    if "buys_cp_12w" in X_train.columns:
        X_train["freq_12w_norm"] = X_train["buys_cp_12w"].astype(float) / 12.0
        X_test["freq_12w_norm"]  = X_test["buys_cp_12w"].astype(float) / 12.0
    if "buys_cp_8w" in X_train.columns:
        X_train["freq_8w_norm"] = X_train["buys_cp_8w"].astype(float) / 8.0
        X_test["freq_8w_norm"]  = X_test["buys_cp_8w"].astype(float) / 8.0

    # Columnas requeridas para el preprocesamiento (ya existentes tras add_deterministic_features)
    required_cols = sorted(list(set(
        cfg["num_base"] + cfg.get("num_extra", []) + cfg["cat_high"] + cfg["cat_low"] + (["recency_bin"] if cfg.get("make_bins", True) else [])
    )))
    missing = [c for c in required_cols if c not in X_train.columns]
    if missing:
        raise KeyError(f"Faltan columnas para preprocesar: {missing}")

    pre = build_preprocessor(cfg)
    clf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=random_state)

    pipe = Pipeline([
        ("preprocess", pre),
        ("model", clf),
    ])

    t0 = time.time()
    pipe.fit(X_train[required_cols], y_train)
    print(f"[fit] pipeline entrenado en {time.time()-t0:.1f}s")

    y_pred = pipe.predict(X_test[required_cols])
    acc = accuracy_score(y_test, y_pred)
    f1_pos = f1_score(y_test, y_pred, pos_label=1)
    print(f"[eval] accuracy (test): {acc:.4f}")
    print(f"[eval] f1-score clase positiva (test): {f1_pos:.4f}")

    out_path = models_dir / model_filename
    joblib.dump(pipe, out_path)
    print(f"[save] pipeline -> {out_path}")

    return str(out_path)




import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import optuna
import numpy as np
import pandas as pd
from optuna.pruners import MedianPruner
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, accuracy_score, f1_score

def optimize_and_train_rf_optuna(
    base_dir: str = "data",
    train_filename: str = "train.parquet",
    valid_filename: str = "valid.parquet",
    test_filename: str  = "test.parquet",
    target_col: str = "y_next_week",
    config: Dict[str, Any] = None,      # usa DEFAULT_CONFIG / build_preprocessor (Opción B)
    n_trials: int = 40,
    random_state: int = 42,
    model_filename: str = "rf_pipeline_optuna.joblib",
    **kwargs,
) -> Tuple[str, Dict[str, Any]]:
    """
    Optimiza SOLO RandomForest con Optuna (métrica: Average Precision en VALID),
    reentrena con TRAIN+VALID y evalúa en TEST (accuracy y F1 de la clase positiva).
    Guarda el pipeline final en data/YYYY-MM-DD/models/{model_filename}.
    Retorna (ruta_modelo, best_params).
    """
    cfg = (config or DEFAULT_CONFIG).copy()

    # === paths (misma lógica de create_folders) ===
    date_str = (
        kwargs.get("ds")
        or (kwargs.get("logical_date").strftime("%Y-%m-%d") if kwargs.get("logical_date") else None)
        or (kwargs.get("execution_date").strftime("%Y-%m-%d") if kwargs.get("execution_date") else None)
    )
    if not date_str:
        raise ValueError("No se encontró la fecha de ejecución en el contexto (ds/logical_date/execution_date).")

    root = Path(base_dir) / date_str
    splits_dir = root / "splits"
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    tr_path = splits_dir / train_filename
    va_path = splits_dir / valid_filename
    te_path = splits_dir / test_filename
    for p in (tr_path, va_path, te_path):
        if not p.exists():
            raise FileNotFoundError(f"Falta split: {p}")

    # === carga (csv o parquet) ===
    def _load_any(p: Path) -> pd.DataFrame:
        suf = p.suffix.lower()
        if suf == ".csv":
            return pd.read_csv(p)
        if suf in (".parquet", ".pq"):
            return pd.read_parquet(p)
        raise ValueError(f"Extensión no soportada: {suf}")

    df_tr = _load_any(tr_path)
    df_va = _load_any(va_path)
    df_te = _load_any(te_path)

    for df, name in [(df_tr, "train"), (df_va, "valid"), (df_te, "test")]:
        if target_col not in df.columns:
            raise ValueError(f"{name}: no contiene la columna objetivo '{target_col}'")

    # Separar X/y y evitar fugas (quitamos llaves y target)
    drop_cols = [c for c in ["week_start", "customer_id", "product_id"] if c in df_tr.columns]
    X_tr, y_tr = df_tr.drop(columns=[target_col] + drop_cols), df_tr[target_col].astype(int).values
    X_va, y_va = df_va.drop(columns=[target_col] + drop_cols), df_va[target_col].astype(int).values
    X_te, y_te = df_te.drop(columns=[target_col] + drop_cols), df_te[target_col].astype(int).values

    # Validaciones mínimas de materia prima
    if "recency_weeks" not in X_tr.columns:
        raise KeyError("Falta 'recency_weeks' para derivar 'days_since_last_cp'.")
    if not any(c.startswith("freq_w_") for c in X_tr.columns):
        raise KeyError("Se necesita al menos una columna 'freq_w_*' (p. ej. 'freq_w_12').")

    # === Features determinísticas ANTES del pipeline ===
    X_tr = add_deterministic_features(X_tr)
    X_va = add_deterministic_features(X_va)
    X_te = add_deterministic_features(X_te)

    # Normalizaciones extra (si las usas en num_extra del config)
    if "buys_cp_12w" in X_tr.columns:
        X_tr["freq_12w_norm"] = X_tr["buys_cp_12w"].astype(float) / 12.0
        X_va["freq_12w_norm"] = X_va["buys_cp_12w"].astype(float) / 12.0
        X_te["freq_12w_norm"] = X_te["buys_cp_12w"].astype(float) / 12.0
    if "buys_cp_8w" in X_tr.columns:
        X_tr["freq_8w_norm"] = X_tr["buys_cp_8w"].astype(float) / 8.0
        X_va["freq_8w_norm"] = X_va["buys_cp_8w"].astype(float) / 8.0
        X_te["freq_8w_norm"] = X_te["buys_cp_8w"].astype(float) / 8.0

    # Columnas requeridas tras crear features determinísticas
    required_cols = sorted(list(set(
        cfg["num_base"] + cfg.get("num_extra", []) + cfg["cat_high"] + cfg["cat_low"] + (["recency_bin"] if cfg.get("make_bins", True) else [])
    )))
    missing_tr = [c for c in required_cols if c not in X_tr.columns]
    if missing_tr:
        raise KeyError(f"Faltan columnas en TRAIN para preprocesar: {missing_tr}")
    missing_va = [c for c in required_cols if c not in X_va.columns]
    if missing_va:
        raise KeyError(f"Faltan columnas en VALID para preprocesar: {missing_va}")
    missing_te = [c for c in required_cols if c not in X_te.columns]
    if missing_te:
        raise KeyError(f"Faltan columnas en TEST para preprocesar: {missing_te}")

    # === Optuna: objetivo = Average Precision en VALID ===
    def objective(trial: optuna.Trial) -> float:
        rf = RandomForestClassifier(
            n_estimators      = trial.suggest_int("n_estimators", 100, 600, step=100),
            max_depth         = trial.suggest_int("max_depth", 8, 32, log=True),
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 10),
            max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            n_jobs=-1,
            random_state=random_state,
        )
        # Preprocesador NUEVO por trial (evita estado cruzado)
        pre = build_preprocessor(cfg)
        pipe = Pipeline([("preprocess", pre), ("model", rf)])
        pipe.fit(X_tr[required_cols], y_tr)
        proba = pipe.predict_proba(X_va[required_cols])[:, 1]
        return average_precision_score(y_va, proba)

    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=10),
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[optuna] best AP(valid)={study.best_value:.4f} | trials={len(study.trials)} | time={time.time()-t0:.1f}s")
    print("[optuna] best params:", json.dumps(study.best_trial.params, indent=2))

    # === Entrena final con TRAIN+VALID y evalúa en TEST ===
    best = study.best_trial.params
    rf_best = RandomForestClassifier(
        n_estimators      = best["n_estimators"],
        max_depth         = best["max_depth"],
        min_samples_split = best["min_samples_split"],
        min_samples_leaf  = best["min_samples_leaf"],
        max_features      = best["max_features"],
        n_jobs=-1,
        random_state=random_state,
    )
    pre_best = build_preprocessor(cfg)
    pipe_best = Pipeline([("preprocess", pre_best), ("model", rf_best)])

    X_trva = pd.concat([X_tr[required_cols], X_va[required_cols]], axis=0)
    y_trva = np.concatenate([y_tr, y_va], axis=0)

    t1 = time.time()
    pipe_best.fit(X_trva, y_trva)
    print(f"[fit] final train+valid = {time.time()-t1:.1f}s")

    y_pred = pipe_best.predict(X_te[required_cols])
    acc = accuracy_score(y_te, y_pred)
    f1_pos = f1_score(y_te, y_pred, pos_label=1)
    print(f"[test] accuracy={acc:.4f} | f1(pos)={f1_pos:.4f}")

    # === Guardar pipeline ===
    out_path = models_dir / model_filename
    joblib.dump(pipe_best, out_path)
    print(f"[save] pipeline -> {out_path}")

    return str(out_path), study.best_trial.params
