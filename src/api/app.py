import sys
import os
import time
from datetime import datetime
from fastapi import Response, HTTPException
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import psutil
import logging
from src.monitoring.monitor import (
    ModelMonitor,
    PREDICTION_COUNTER,
    PREDICTION_LATENCY,
    ERROR_COUNTER,
    ACTIVE_REQUESTS,
)
from src.api.main import app, model, PatientData


# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Initialize monitor
monitor = ModelMonitor()


# Add metrics endpoint for Prometheus (Grafana data source)
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Enhanced health endpoint with Grafana-friendly metrics
@app.get("/health")
@app.get("/render-health")
async def render_health():
    """Enhanced health check for Render monitoring"""
    health_info = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "fever-monitoring-api",
        "model_version": "v20",
        "model_path": "models/production/current",
        "monitoring": "enabled",
        "grafana_ready": True,
        "checks": {
            "model_loaded": model is not None,  # FIXED: is not None
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "mlflow_available": os.path.exists("mlruns"),
            "current_model": os.path.exists("models/production/current"),
        },
    }
    return health_info


# Add monitoring to your existing predict endpoint
@app.post("/predict")
async def predict(features: PatientData):
    ACTIVE_REQUESTS.inc()
    start_time = time.time()

    try:
        # Your existing prediction logic here
        prediction = model.predict(
            [
                features.temperature,
                features.age,
                features.bmi,
                features.humidity,
                features.aqi,
                features.heart_rate,
                features.gender,
                features.headache,
                features.body_ache,
                features.fatigue,
                features.chronic_conditions,
                features.allergies,
                features.smoking_history,
                features.alcohol_consumption,
                features.physical_activity,
                features.diet_type,
                features.blood_pressure,
                features.previous_medication,
                features.recommended_medication,
            ]
        )

        probability = model.predict_proba(
            [
                features.temperature,
                features.age,
                features.bmi,
                features.humidity,
                features.aqi,
                features.heart_rate,
                features.gender,
                features.headache,
                features.body_ache,
                features.fatigue,
                features.chronic_conditions,
                features.allergies,
                features.smoking_history,
                features.alcohol_consumption,
                features.physical_activity,
                features.diet_type,
                features.blood_pressure,
                features.previous_medication,
                features.recommended_medication,
            ]
        )

        latency = time.time() - start_time

        # Log prediction for monitoring
        monitor.log_prediction(
            features=features.dict(), prediction=prediction[0], latency=latency
        )

        ACTIVE_REQUESTS.dec()

        # Convert prediction to risk level for Streamlit
        prediction_value = int(prediction[0])
        prob_value = float(probability[0][1])

        if prediction_value == 1:
            risk_level = "High"
        elif prediction_value == 0:
            risk_level = "Low"
        else:
            risk_level = "Medium"

        return {
            "risk_level": risk_level,
            "prediction": prediction_value,
            "probability": prob_value,
            "latency_ms": round(latency * 1000, 2),
            "model_version": "v20",
            "confidence": "High" if prob_value > 0.8 else "Medium",
        }

    except Exception as e:
        ERROR_COUNTER.inc()
        ACTIVE_REQUESTS.dec()
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")
