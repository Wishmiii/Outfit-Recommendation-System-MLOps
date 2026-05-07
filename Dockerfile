FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api

COPY mlflow.db ./mlflow.db
COPY mlruns ./mlruns

# Make the Windows MLflow artifact path work inside Linux Docker
RUN mkdir -p "/C:/closet_mlops" && ln -s /app/mlruns "/C:/closet_mlops/mlruns"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]