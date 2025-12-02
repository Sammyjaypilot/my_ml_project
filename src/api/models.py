from pydantic import BaseModel, Field
from typing import Optional

# Request model for prediction endpoint
class PatientData(BaseModel):
    temperature: float = Field(example=38.5)
    age: int = Field(example=35)
    bmi: float = Field(example=24.2)
    humidity: float = Field(example=65.0)
    aqi: float = Field(example=45.0)
    heart_rate: float = Field(example=85.0)
    gender: str = Field(example="Male")
    headache: str = Field(example="Yes")
    body_ache: str = Field(example="No")
    fatigue: str = Field(example="Yes")
    chronic_conditions: str = Field(example="None")
    allergies: str = Field(example="No")
    smoking_history: str = Field(example="Non-smoker")
    alcohol_consumption: str = Field(example="Occasional")
    physical_activity: str = Field(example="Moderate")
    diet_type: str = Field(example="Balanced")
    blood_pressure: str = Field(example="Normal")
    previous_medication: str = Field(example="None")
    recommended_medication: str = Field(example="Paracetamol")


# Response model for prediction endpoint
class PredictionResponse(BaseModel):
    prediction: int = Field(example=1)
    probability: float = Field(example=0.85)
    risk_level: str = Field(example="High")


# Health check response model
class HealthResponse(BaseModel):
    status: str = Field(example="healthy")
    message: Optional[str] = Field(default=None, example="API is running normally")


# Basic API response model
class APIResponse(BaseModel):
    message: str = Field(example="FeverSeverity Prediction API is running")
    status: str = Field(default="success", example="success")


    