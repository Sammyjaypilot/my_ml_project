from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np
import json
import os
import yaml
import logging
from datetime import datetime 
import mlflow
import joblib  # Add this for model saving
from pathlib import Path

# Fix imports - use absolute paths or direct imports
try:
    from ml.data_processor import DataProcessor, load_params, get_data_preprocessor
except ImportError:
    from data_processor import DataProcessor, load_params, get_data_preprocessor

# Import model classes directly instead of relying on Model class
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

def filter_valid_params(estimator_class, params):
    """Keep only parameters that are valid for this sklearn estimator."""
    if not params:
        return {}
    try:
        valid_keys = estimator_class().get_params().keys()
        return {k: v for k, v in params.items() if k in valid_keys}
    except Exception as e:
        print(f"⚠️ filter_valid_params error: {e}")
        return {}

class TrainingPipeline:
    def __init__(self, preprocessor, model_type=None, random_state=None, model_params=None, params_yaml_path=None):
        """
        Simplified initialization compatible with your existing structure
        """
        self.preprocessor = preprocessor
        self.model_type = model_type
        self.random_state = random_state or 42
        self.model_params = model_params or {}
        self.pipeline = None
        self.training_history = {}
        self.evaluation_results = {}
        
        # Use your existing params.yaml path
        self.params_yaml_path = Path("C:/Users/DELL/Desktop/my_ml_project/notebook/params.yaml")
        
        # Load parameters from your existing params.yaml
        self._load_parameters()
        
        # MLflow setup
        try:
            mlflow.sklearn.autolog(disable=True)
        except Exception:
            pass

    def _load_parameters(self):
        """Load parameters from your existing params.yaml structure"""
        try:
            if self.params_yaml_path.exists():
                params = load_params(self.params_yaml_path)
                model_cfg = params.get('model', {})
                
                # Set model_type if not provided
                if self.model_type is None:
                    self.model_type = model_cfg.get('model_type', 'random_forest')
                
                # Set random_state if not provided
                if self.random_state is None:
                    self.random_state = model_cfg.get('random_state', 42)
                
                # Load model-specific parameters
                if not self.model_params:
                    if self.model_type in model_cfg:
                        self.model_params = model_cfg.get(self.model_type, {})
                    
                print(f"✅ Loaded parameters for {self.model_type}: {list(self.model_params.keys())}")
                
        except Exception as e:
            logger.warning(f"Could not read params.yaml: {e}. Using provided defaults.")

    def _create_pipeline(self):
        """Create sklearn pipeline with the specified model"""
        if self.model_type is None:
            raise ValueError("model_type must be specified")
        
        # Map model types to classifier classes
        MODEL_MAP = {
            "random_forest": RandomForestClassifier,
            "logistic_regression": LogisticRegression,
            "xgboost": XGBClassifier,
            "svm": SVC,
            "knn": KNeighborsClassifier,
            "decision_tree": DecisionTreeClassifier
        }
        
        if self.model_type not in MODEL_MAP:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        
        # Filter valid parameters
        classifier_class = MODEL_MAP[self.model_type]
        valid_params = filter_valid_params(classifier_class, self.model_params)
        
        # Add random_state if the classifier supports it
        if 'random_state' in classifier_class().get_params():
            valid_params['random_state'] = self.random_state
        
        # Create classifier
        classifier = classifier_class(**valid_params)
        
        # Create pipeline
        self.pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("classifier", classifier)
        ])
        
        print(f"✅ Created pipeline with {self.model_type} classifier")

    def fit(self, X, y, experiment_name="FeverSeverity_Prediction", run_name=None):
        """Train the model with MLflow tracking"""
        if self.pipeline is None:
            self._create_pipeline()

        # MLflow setup - use your existing tracking URI
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment(experiment_name)

        if run_name is None:
            run_name = f"{self.model_type}_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        start_time = datetime.now()
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            self.training_history["mlflow_run_id"] = run_id

            # Log parameters
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("random_state", self.random_state)
            if self.model_params:
                mlflow.log_params(self.model_params)

            # Train model
            print(f"🔁 Training {self.model_type} (run_id={run_id})...")
            self.pipeline.fit(X, y)
            training_time = (datetime.now() - start_time).total_seconds()

            # Store training history
            self.training_history.update({
                "training_time_seconds": training_time,
                "training_samples": len(X),
                "features_count": X.shape[1],
                "model_type": self.model_type,
                "random_state": self.random_state,
                "training_timestamp": datetime.now().isoformat()
            })

            # Log to MLflow
            mlflow.log_metric("training_time", training_time)
            mlflow.log_param("feature_count", X.shape[1])
            mlflow.sklearn.log_model(self.pipeline, "model")

            print(f"✅ Training completed in {training_time:.2f}s")

        return self

    # Keep all your existing evaluation methods (they're good!)
    def predict(self, X):
        if self.pipeline is None:
            raise ValueError("Model must be trained first")
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        if self.pipeline is None:
            raise ValueError("Model must be trained first")
        return self.pipeline.predict_proba(X)

    def evaluate(self, X, y, dataset_name='validation'):
        # ... [keep your existing evaluate method] ...
        pass

    def get_comprehensive_report(self, X_train, y_train, X_val, y_val):
        # ... [keep your existing method] ...
        pass

    def save_training_report(self, report, filepath="reports/training_metrics.json"):
        # ... [keep your existing method] ...
        pass

    def save_model(self, filepath):
        """Save the trained pipeline using joblib"""
        if self.pipeline is None:
            raise ValueError("No trained model to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.pipeline, filepath)
        print(f"✅ Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained pipeline"""
        self.pipeline = joblib.load(filepath)
        print(f"✅ Model loaded from {filepath}")

    def register_model(self, model_name, transition_to_stage="Staging"):
        """Register model in MLflow Model Registry"""
        run_id = self.training_history.get("mlflow_run_id")
        if not run_id:
            raise ValueError("No MLflow run ID found")

        client = MlflowClient()
        model_uri = f"runs:/{run_id}/model"

        try:
            # Create registered model if it does not exist
            try:
                client.create_registered_model(model_name)
                print(f"📝 Created new registered model: {model_name}")

            except Exception:
                print(f"📝 Using existing registered model: {model_name}")

            model_version = client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id
            )

            print(f"📦 Model version {model_version.version} registered")

            if transition_to_stage:
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage=transition_to_stage
                )
                print(f"🚀 Transitioned to {transition_to_stage}")

            return model_version
        except Exception as e:
            print(f"❌ Model registration failed: {e}")
            # Don't re-raise the exception, just log it and continue
            return None
        
        