import mlflow
import pandas as pd
import logging
from datetime import datetime
import prometheus_client as prom
from prometheus_client import Counter, Histogram, Gauge
import os

# Initialize Prometheus metrics for Grafana
PREDICTION_COUNTER = Counter("model_predictions_total", "Total predictions made")
PREDICTION_LATENCY = Histogram("model_prediction_latency_seconds", "Prediction latency")
MODEL_ACCURACY = Gauge("model_accuracy", "Current model accuracy")
ERROR_COUNTER = Counter("model_errors_total", "Total prediction errors")
ACTIVE_REQUESTS = Gauge("active_requests", "Currently active requests")


class ModelMonitor:
    def __init__(self):
        # Use existing MLflow database
        self.setup_mlflow_experiment()

    def setup_mlflow_experiment(self):
        mlflow.set_experiment("feverseverity-model-production")

    def log_prediction(self, features, prediction, actual=None, latency=None):
        # Log to existing MLflow setup
        with mlflow.start_run():
            mlflow.log_param("prediction_timestamp", datetime.now())
            mlflow.log_dict(features, "input_features.json")
            mlflow.log_metric("prediction_value", float(prediction))

            if latency:
                mlflow.log_metric("prediction_latency_ms", latency * 1000)
                PREDICTION_LATENCY.observe(latency)

            if actual is not None:
                mlflow.log_metric("actual_value", float(actual))
                accuracy = 1.0 if prediction == actual else 0.0
                MODEL_ACCURACY.set(accuracy)

        # Increment Prometheus counter for Grafana
        PREDICTION_COUNTER.inc()
