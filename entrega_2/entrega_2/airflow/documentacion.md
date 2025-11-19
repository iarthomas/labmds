* **Documentación detallada:**  

  # Descripción del `DAG` (funcionalidades de cada tarea y cómo se relacionan entre sí).

El DAG se llama ml_pipeline_manual, no tiene schedule (se ejecuta manualmente) y no hace catchup.

En el DAG, primero se configuran rutas y nombres de archivos:

RAW_DATA_DIR = "/home/y/airflow/parquet"  donde están:

clientes.parquet

productos.parquet

transacciones.parquet

BASE_DIR = "/home/y/airflow/data" donde se crean carpetas por fecha ({ds}/raw, {ds}/splits, {ds}/models). 

## Tarea 1: start (EmptyOperator)

Solo marca el inicio del flujo. Sirve para dejar claro que todo lo que viene después va desde este punto.

## Tarea 2: load_raw_data → _load_raw_data

Obtiene ds del contexto de Airflow (formato YYYY-MM-DD).

Lee desde RAW_DATA_DIR:

clientes.parquet

productos.parquet

transacciones.parquet

Llama a build_from_dataframes(...) (de pipeline_lib) para:

### Limpiar y tipificar:

- clientes (customer_id, region_id, customer_type, etc.),

- productos (product_id, brand, category, etc.),

- transacciones (purchase_date, payment, etc.) con manejo de zona horaria (se usa "America/Santiago" en el DAG). 

### Construir un dataset supervisado a nivel semana–cliente–producto que incluye, entre otros:

- Agregaciones semanales: has_purchase, n_purchases, pay_sum.

- Ventanas históricas de compra: freq_w_12, pay_sum_12 (y en general freq_w_{lookback}, pay_sum_{lookback}).

- Índice temporal y recencia: week_idx, recency_weeks.

- Target: y_next_week (si compra la próxima semana o no).

- Merge con atributos de cliente y producto (región, tipo de cliente, marca, categoría, etc.).

### Llama a save_supervised_to_raw(...) para:

- Crear carpeta BASE_DIR/{ds}/raw.

- Guardar supervised.parquet ahí. 


## Tarea 3: split_temporal → _split_temporal

### Lee BASE_DIR/{ds}/raw/supervised.parquet.

### Llama a split_supervised_temporal_holdout(...) con:

- date_col="week_start".

- train_ratio=0.7, valid_ratio=0.15 (el 15 % restante es test).

- gap_weeks=0 (sin hueco entre conjuntos). 

### Hace un split temporal por semanas:

- Se ordenan las semanas únicas.

- Train = primeras ~70 % de semanas.

- Valid = siguientes ~15 %.

- Test = resto.

- Se controla que no haya solapamiento entre rangos.

###  Guarda:

- train.parquet, valid.parquet, test.parquet en BASE_DIR/{ds}/splits/. 

## Tarea 4: train_model → _train_model

### Llama a preprocess_and_train(...) leyendo:

- train.parquet y test.parquet desde BASE_DIR/{ds}/splits/.

- Usa como target y_next_week. 

### Dentro de preprocess_and_train:

- Se eliminan columnas llave (week_start, customer_id, product_id) para evitar fugas.

### Se crean features determinísticas adicionales con add_deterministic_features:

- days_since_last_cp (días desde la última compra).

- buys_cp_12w, buys_cp_8w, buys_cp_4w.

- activity_intensity (visitas + deliveries).

- recency_inverse, recent_7d, recent_14d.

- recency_bin (bines de recencia). 


### Se derivan normalizaciones como freq_12w_norm, freq_8w_norm (si existen). 


### Se arma un preprocesador (build_preprocessor):

- Numéricas: imputación + RobustScaler (por defecto).

- Categóricas “high-card” (brand, region_id): se les agrega freq-encoding.

- Categóricas “low-card” (category, sub_category, segment, package, customer_type, recency_bin): One-Hot Encoding. 

### Se entrena un RandomForestClassifier (200 árboles, etc.) en TRAIN.

### Se evalúa en TEST (accuracy y F1 de la clase positiva).

### Se guarda el pipeline completo (preprocesamiento + modelo) en:

- BASE_DIR/{ds}/models/rf_pipeline.joblib. 


## Tarea 5: optimize_model → _optimize_model

### Llama a optimize_and_train_rf_optuna(...) leyendo:

- train.parquet, valid.parquet, test.parquet desde BASE_DIR/{ds}/splits/. 

### Repite la lógica de features determinísticas y preprocesador, pero ahora:

- Usa Optuna para optimizar SOLO los hiperparámetros del Random Forest:

- n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, etc.

### Métrica objetivo: Average Precision en VALID. 

### Flujo de optimización:

- Para cada trial: Construye un RandomForestClassifier con hiperparámetros propuestos.

- Construye un preprocesador nuevo.

- Entrena en TRAIN y evalúa AP en VALID.

- Optuna selecciona los mejores hiperparámetros. 


### Modelo final:

- Se reentrena un pipeline con los mejores hiperparámetros usando TRAIN+VALID.

- Se evalúa en TEST (accuracy y F1).

- Se guarda en: BASE_DIR/{ds}/models/rf_pipeline_optuna.joblib. 


### Tarea 6: done (EmptyOperator)

Marca el final del DAG, una vez terminada la optimización del modelo.



OJO que las dependencias son completamente lineales (no se completan tareas en paralelo):

start 
  → load_raw_data 
  → split_temporal 
  → train_model 
  → optimize_model 
  → done



  # Diagrama de flujo del *pipeline* completo.


![alt text](image.png)


  # Representación visual del `DAG` en la interfaz de `Airflow`.




![alt text](image-1.png)




  # Explicación de cómo se diseñó la lógica para integrar futuros datos, y reentrenar el modelo.


La lógica para integrar datos futuros y reentrenar el modelo se basa en que cada vez que llegan nuevas semanas de información se actualizan los `parquet` crudos de clientes, productos y transacciones, y luego se ejecuta el DAG indicando una fecha (`ds`), lo que dispara un rebuild completo: primero se vuelve a construir el dataset supervisado usando toda la historia disponible y se guarda en `data/{ds}/raw/supervised.parquet`, después se hace un split temporal en train/valid/test que siempre se ajusta automáticamente a las últimas semanas (el “futuro” queda en el test más reciente), guardando esos conjuntos en `data/{ds}/splits/`, finalmente, con esos splits, se reentrena desde cero un modelo baseline y otro optimizado con Optuna, ambos empaquetados como *pipelines* de preprocesamiento + Random Forest y versionados en `data/{ds}/models/` (por ejemplo `rf_pipeline.joblib` y `rf_pipeline_optuna.joblib`). Toda la estructura depende sistemáticamente de la fecha de ejecución, de modo que cada corrida genera una carpeta nueva aislada y reproducible con sus datos procesados y sus modelos reentrenados, sin pisar ni reutilizar artefactos de ejecuciones anteriores.
