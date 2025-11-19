FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py ./main.py
COPY pipeline_lib.py ./pipeline_lib.py
ENV BASE_DIR=/models MODEL_FILENAME=rf_pipeline_optuna.joblib THRESHOLD=0.5
EXPOSE 8000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
