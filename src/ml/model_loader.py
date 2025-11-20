import mlflow.pyfunc
import joblib
import os
import json
from datetime import datetime

def load_production_model():
    """
    Load the production model from MLflow registry or local file.
    """
    model = None

    # Strategy 1: Try MLflow Model Registry
    try:
        # Try to load from MLflow first
        model = mlflow.pyfunc.load_model("models:/FeverSeverityPrediction/Production")
        print("✅ Loaded model from MLflow Registry")

        return model
    
    except Exception as e:
        print(f"MLflow Registry load failed: {e}")  # FIXED: single braces

    # Strategy 2: Try local MLflow model
    try:
        model = mlflow.pyfunc.load_model("mlruns/0/latest/model")
        print("✅ Loaded model from local MLflow")
        return model
    except Exception as e:
        print(f"Local MLflow load failed: {e}")  # FIXED: single braces
    
    # Strategy 3: Fallback to local pickle file
    try:
        model_path = os.path.join(os.path.dirname(__file__), '../../models/fever_severity_model.pkl')
        model = joblib.load(model_path)
        print("✅ Loaded model from local pickle file")
        return model
    except Exception as e:
        print(f"Local pickle load failed: {e}")  # FIXED: single braces
        raise RuntimeError("No model could be loaded")
    
def get_model_info():
    """Get information about the loaded model"""
    return {  # FIXED: single braces
        "version": "2.0.0",
        "model_type": "RandomForest",
        "training_date": "2024-01-10",
        "performance": {  # FIXED: single braces
            "accuracy": 0.85,
            "roc_auc": 0.92,
            "precision": 0.83,
            "recall": 0.81
        }
    }