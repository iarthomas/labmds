# dags/ml_pipeline_dag.py
from datetime import datetime
from pathlib import Path

import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator, get_current_context
from airflow.operators.empty import EmptyOperator  # Airflow 2.x

#  IMPORTA .py 

from pipeline_lib import (
    build_from_dataframes,
    save_supervised_to_raw,
    split_supervised_temporal_holdout,
    preprocess_and_train,
    optimize_and_train_rf_optuna,
)

# === CONFIG EDITABLE ===
RAW_DATA_DIR = "/home/y/airflow/parquet"   # carpeta con clientes/productos/transacciones
CLIENTES_FILE = "clientes.parquet"
PRODUCTOS_FILE = "productos.parquet"
TRANSACCIONES_FILE = "transacciones.parquet"

BASE_DIR = "/home/y/airflow/data"          # aquí se crean {ds}/raw|splits|models



# ---------- Wrappers de tareas (reciben ds desde el contexto) ----------
def _load_raw_data():
    ctx = get_current_context()
    ds = ctx["ds"]  # 'YYYY-MM-DD'

    input_dir = Path(RAW_DATA_DIR)
    df_clientes = pd.read_parquet(input_dir / CLIENTES_FILE)
    df_productos = pd.read_parquet(input_dir / PRODUCTOS_FILE)
    df_tx = pd.read_parquet(input_dir / TRANSACCIONES_FILE)

    # Construye dataset supervisado y guárdalo en data/{ds}/raw/supervised.parquet
    ds_df = build_from_dataframes(
        df_clientes=df_clientes,
        df_productos=df_productos,
        df_transacciones=df_tx,
        lookback_weeks=12,
        top_region_type=20,
        tz="America/Santiago",
    )
    save_supervised_to_raw(ds_df, base_dir=BASE_DIR, filename="supervised.parquet", ds=ds)


def _split_temporal():
    ctx = get_current_context()
    ds = ctx["ds"]
    split_supervised_temporal_holdout(
        base_dir=BASE_DIR,
        supervised_filename="supervised.parquet",
        date_col="week_start",
        train_ratio=0.7,
        valid_ratio=0.15,
        gap_weeks=0,
        ds=ds,
    )


def _train_model():
    ctx = get_current_context()
    ds = ctx["ds"]
    preprocess_and_train(
        base_dir=BASE_DIR,
        train_filename="train.parquet",
        test_filename="test.parquet",
        target_col="y_next_week",
        model_filename="rf_pipeline.joblib",
        ds=ds,
    )


def _optimize_model():
    ctx = get_current_context()
    ds = ctx["ds"]
    optimize_and_train_rf_optuna(
        base_dir=BASE_DIR,
        train_filename="train.parquet",
        valid_filename="valid.parquet",
        test_filename="test.parquet",
        target_col="y_next_week",
        n_trials=40,
        model_filename="rf_pipeline_optuna.joblib",
        ds=ds,
    )


# ---------------- DAG (ejecución manual) ----------------
with DAG(
    dag_id="ml_pipeline_manual",
    description="Build supervised → split temporal → train → Optuna; ejecución manual",
    start_date=datetime(2023, 1, 1),
    schedule=None,   # sin schedule; se dispara manualmente
    catchup=False,
    default_args={"owner": "airflow"},
    tags=["ml", "manual", "training"],
) as dag:

    start = EmptyOperator(task_id="start")

    load_raw_data = PythonOperator(
        task_id="load_raw_data",
        python_callable=_load_raw_data,
    )

    split_temporal = PythonOperator(
        task_id="split_temporal",
        python_callable=_split_temporal,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
    )

    optimize_model = PythonOperator(
        task_id="optimize_model",
        python_callable=_optimize_model,
    )

    done = EmptyOperator(task_id="done")

    # Secuencia
    start >> load_raw_data >> split_temporal >> train_model >> optimize_model >> done
