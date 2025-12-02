from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import sys
import os
import traceback
from pydantic import BaseModel, Field
import logging
from typing import Optional
from datetime import datetime
import json
from fastapi.responses import JSONResponse


# Add src to path to import your ML modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.ml.model_loader import load_production_model, get_model_info

from .models import PatientData, PredictionResponse, HealthResponse, APIResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="FeverSeverity Prediction API",
    version="1.00",
    description="API for predicting fever severity based on patient data",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Load your production model
try:
    model = load_production_model()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

model_info = get_model_info()


# Enhanced request/response models
class PatientData(BaseModel):
    temperature: float = Field(
        ..., ge=0, le=300, description="Body temperature in Fahrenheit"
    )
    age: int = Field(..., ge=0, le=120, description="Age of the patient in years")
    bmi: float = Field(..., ge=0, le=100, description="Body Mass Index")
    humidity: float = Field(
        ..., ge=0, le=100, description="Environmental humidity percentage"
    )
    aqi: int = Field(..., ge=0, le=500, description="Air Quality Index")
    heart_rate: int = Field(
        ..., ge=0, le=220, description="Heart rate in beats per minute"
    )
    gender: str = Field(..., description="Patient gender (Male/Female/Other)")
    headache: str = Field(..., description="Presence of headache (Yes/No)")
    body_ache: str = Field(..., description="Presence of body ache (Yes/No)")
    fatigue: str = Field(..., description="Presence of fatigue (Yes/No)")
    chronic_conditions: str = Field(
        ..., description="Presence of chronic conditions (Yes/No/None)"
    )
    allergies: str = Field(..., description="Presence of allergies (Yes/No)")
    smoking_history: str = Field(
        ..., description="Smoking history (Non-smoker/Former/Current)"
    )
    alcohol_consumption: str = Field(
        ..., description="Alcohol consumption (None/Occasional/Regular)"
    )
    physical_activity: str = Field(
        ..., description="Physical activity level (Low/Moderate/High)"
    )
    diet_type: str = Field(
        ..., description="Diet type (Balanced/Vegetarian/Vegan/Other)"
    )
    blood_pressure: str = Field(
        ..., description="Blood pressure category (Normal/High/Low)"
    )
    previous_medication: str = Field(
        ..., description="Previous medication usage (Yes/No/None)"
    )
    recommended_medication: str = Field(
        ..., description="Recommended medication (Paracetamol/Ibuprofen/None)"
    )


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    model_version: str
    confidence: str


class ModelInfo(BaseModel):
    version: str
    model_type: str
    training_date: str
    performance: dict


@app.get("/")
async def root():
    return {"message": "FeverSeverity Prediction API v2.0", "status": "operational"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_information():
    """Get information about the currently loaded model."""
    return model_info


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient_data: PatientData):
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        logger.info(
            f"Received prediction request for patient: {{patient_data.age}} years"
        )

        # Define column names
        feature_columns = [
            "Temperature",
            "Age",
            "BMI",
            "Humidity",
            "AQI",
            "Heart_Rate",
            "Gender",
            "Headache",
            "Body_Ache",
            "Fatigue",
            "Chronic_Conditions",
            "Allergies",
            "Smoking_History",
            "Alcohol_Consumption",
            "Physical_Activity",
            "Diet_Type",
            "Blood_Pressure",
            "Previous_Medication",
            "Recommended_Medication",
        ]

        # Convert patient data to features array
        features = [
            patient_data.temperature,  # Temperature
            patient_data.age,  # Age
            patient_data.bmi,  # BMI
            patient_data.humidity,  # Humidity
            patient_data.aqi,  # AQI
            patient_data.heart_rate,  # Heart_Rate
            patient_data.gender,  # Gender
            patient_data.headache,  # Headache
            patient_data.body_ache,  # Body_Ache
            patient_data.fatigue,  # Fatigue
            patient_data.chronic_conditions,  # Chronic_Conditions
            patient_data.allergies,  # Allergies
            patient_data.smoking_history,  # Smoking_History
            patient_data.alcohol_consumption,  # Alcohol_Consumption
            patient_data.physical_activity,  # Physical_Activity
            patient_data.diet_type,  # Diet_Type
            patient_data.blood_pressure,  # Blood_Pressure
            patient_data.previous_medication,  # Previous_Medication
            patient_data.recommended_medication,  # Recommended_Medication
        ]

        print(f"Features: {features}")  # Debug print

        # Create DataFrame
        features_df = pd.DataFrame([features], columns=feature_columns)
        print(f"DataFrame created: {features_df.shape}")  # Debug print

        # Make prediction
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        probability = probabilities[1]  # for positive class

        # show risk assessment
        # Enhanced risk assessment
        if probability >= 0.7:
            risk_level = "High"
            confidence = "High"
        elif probability >= 0.4:
            risk_level = "Medium"
            confidence = "Medium"
        else:
            risk_level = "Low"
            confidence = "High"

        logger.info(
            f"Prediction completed: {{risk_level}} risk (probability: {{probability:.3f}})"
        )

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            model_version=model_info["version"],
            confidence=confidence,
        )

    except Exception as e:
        logger.error(f"Prediction error: {{str(e)}}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {{str(e)}}")


@app.get("/patients/{{patient_id}}/risk")
async def get_patient_risk(
    patient_id: int,
    include_details: bool = Query(False, description="Include detailed analysis"),
):
    """Get risk assessment for a specific patient ID"""

    print(
        f"🔍 Debug: Patient risk endpoint called for patient_id={patient_id}, include_details={include_details}"
    )

    try:
        # For demonstration, we will mock the prediction
        mock_patient_data = PatientData(
            temperature=38.5,
            age=35,
            bmi=24.2,
            humidity=65.0,
            aqi=45,
            heart_rate=85,
            gender="Male",
            headache="Yes",
            body_ache="No",
            fatigue="Yes",
            chronic_conditions="None",
            allergies="No",
            smoking_history="Non-smoker",
            alcohol_consumption="Occasional",
            physical_activity="Moderate",
            diet_type="Balanced",
            blood_pressure="Normal",
            previous_medication="None",
            recommended_medication="Paracetamol",
        )

        print("✅ Mock patient data created successfully")

        prediction = await predict(mock_patient_data)
        print("✅ Prediction completed")

        response = {"patient_id": patient_id, "risk_assessment": prediction}

        if include_details:
            response["details"] = {
                "last_checkup": "2025-11-11",
                "history": "No significant history",
            }

        print(f"✅ Response prepared: {response}")
        return response

    except Exception as e:
        print(f"❌ Error in patient risk endpoint: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Patient risk assessment failed: {str(e)}"
        )


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.warning(f"Value error: {{str(exc)}}")
    return JSONResponse(
        status_code=400, content={{"detail": f"Invalid input data: {{str(exc)}}"}}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
