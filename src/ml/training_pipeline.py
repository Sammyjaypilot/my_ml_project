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
from .data_processor import get_data_preprocessor
from .model import Model
from pathlib import Path
from .data_processor import DataProcessor


data_processor = DataProcessor()
preprocessor = data_processor.create_preprocessor()

logger = logging.getLogger(__name__)
class TrainingPipeline:
    def __init__(self, preprocessor=preprocessor, model_type=None, random_state=None, **model_params):
        
        #params_file_path = "C:/Users/DELL/Desktop/my_ml_project/params.yaml"
        current_file = Path(__file__)
        params_file_path = current_file.parent.parent.parent / 'params.yaml'
        print(f"DEBUG: Attempting to open file at: {params_file_path}") # Verify path one last time
        
        try:
            with open(params_file_path, 'r') as f:
                params = yaml.safe_load(f)
            
            self.model_type = model_type or params['model']['model_type']
            self.random_state = random_state or params['model']['random_state']
            self.model_params = model_params or params['model'].get(self.model_type, {})
            print("DEBUG: Successfully loaded parameters.")
            

        except FileNotFoundError as e:
            print(f"FATAL ERROR: Could not find the file at the expected location.")
            print("ACTION REQUIRED: Ensure 'params.yaml' is in your current working directory.")
            raise e
        
            
        except Exception as e:
            print(f"FATAL ERROR: An unexpected issue occurred: {e}")
            self.model_type = model_type or 'random_forest'
            self.random_state = random_state or 42
            self.model_params = model_params or {}


        

        #try:
        #    #mlflow.sklearn.autolog()
        #    self.mlflow_autolog_enabled = True  # Specific to autologging
        #    logger.info("MLflow autologging enabled successfully")
        #except Exception as e:
        #    self.mlflow_autolog_enabled = False
        #    logger.warning(f"MLflow autologging failed: {str(e)}. Manual logging still available.")
        try:
            mlflow.sklearn.autolog(disable=True)  # Disable here too
        except:
            pass

        self.preprocessor = preprocessor
        self.model = Model(
            model_type=self.model_type, 
            random_state=self.random_state, 
            **self.model_params
        )
        self.pipeline = None
        self.training_history = {}
        self.evaluation_results = {}
        self._create_pipeline()

    def _create_pipeline(self):
        """
        create the pipeline with preprocessor and classifier
        """
        if self.model.classifier is None:
            raise ValueError("Model classifier is not initialised")
        
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', self.model.classifier)
        ])
        print(f"✅ Created pipeline with {self.model_type} classifier")

    def fit(self, X, y):
        """
        Train the model and track training history
        """
        if self.pipeline is None:
            self._create_pipeline()
            
        start_time = datetime.now()
     
        self.pipeline.fit(X, y)

        training_time = (datetime.now() - start_time).total_seconds()

        self.training_history = {
            'training_time_seconds': training_time,
            'training_samples': len(X),
            'features_count': X.shape[1],
            'model_type': self.model_type,
            'random_state': self.random_state,
            'training_timestamp': datetime.now().isoformat()
        }

        #try:
        #    mlflow.log_metric("training_time", training_time)
        #    mlflow.log_param("feature_count", X.shape[1])
        #    mlflow.log_param("model_type", self.model_type)
        #    
        #    if hasattr(self, 'pipeline') and self.pipeline is not None:
        #        mlflow.sklearn.log_model(self.pipeline, "model")
                
        #except Exception as e:
        #    logger.warning(f"Manual MLflow logging failed: {str(e)}")

        print(f"✅ Training completed in {training_time:.2f} seconds")

        return self

    def predict(self, X):
        """Make predictions"""
        if self.pipeline is None:
            raise ValueError("Model must be trained before making predictions")
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.pipeline is None:
            raise ValueError("Model must be trained before making predictions")
        return self.pipeline.predict_proba(X)

    def evaluate(self, X, y, dataset_name='validation'):
        """
        Comprehensive evaluation on a dataset
        
        Args:
            X: Features
            y: True labels
            dataset_name: Name of the dataset for reporting
            
        Returns:
            dict: Comprehensive evaluation metrics
        """
        if self.pipeline is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        
        # metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0),
        }
        
        # ROC-AUC 
        if len(np.unique(y)) == 2:
            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba[:, 1])
        else:
            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba, multi_class='ovr', average='weighted')
        
        cm = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        metrics['class_distribution'] = {
            'class_0': int((y == 0).sum()),
            'class_1': int((y == 1).sum()),
            'class_0_percentage': float((y == 0).mean() * 100),
            'class_1_percentage': float((y == 1).mean() * 100)
        }
        
        self.evaluation_results[dataset_name] = metrics
        
        return metrics
    
    def get_comprehensive_report(self, X_train, y_train, X_val, y_val):
        """
        Generate comprehensive training and validation report
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            dict: Comprehensive report with training and validation metrics
        """
        
        train_metrics = self.evaluate(X_train, y_train, 'training')
        val_metrics = self.evaluate(X_val, y_val, 'validation')
        
        # Calculate overfitting indicators
        accuracy_gap = train_metrics['accuracy'] - val_metrics['accuracy']
        f1_gap = train_metrics['f1'] - val_metrics['f1']
        
        comprehensive_report = {
            'training_metadata': self.training_history,
            'training_metrics': train_metrics,
            'validation_metrics': val_metrics,
            'performance_analysis': {
                'accuracy_gap': accuracy_gap,
                'f1_gap': f1_gap,
                'is_overfitting': accuracy_gap > 0.1,  # More than 10% gap
                'is_underfitting': train_metrics['accuracy'] < 0.7,  # Poor training performance
                'generalization_quality': 'good' if accuracy_gap < 0.05 else 'moderate' if accuracy_gap < 0.1 else 'poor'
            },
            'model_summary': {
                'model_type': self.model_type,
                'best_metric': 'accuracy',
                'best_score': val_metrics['accuracy'],
                'training_status': 'completed'
            }
        }
        # Log model performance characteristics
        mlflow.set_tag("generalization_quality", comprehensive_report['performance_analysis']['generalization_quality'])
        mlflow.set_tag("is_overfitting", str(comprehensive_report['performance_analysis']['is_overfitting']))

        return comprehensive_report
    
    def save_training_report(self, report, filepath="reports/training_metrics.json"):
        """Save training report to JSON file"""
        import json
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_report = convert_to_serializable(report)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        print(f"✅ Training report saved to {filepath}")
        return filepath
    
    def save_model(self, filepath):
        """Save the trained pipeline"""
        if self.pipeline is None:
            raise ValueError("No trained model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save using the model's save method
        self.model.pipeline = self.pipeline
        self.model.save(filepath)
        
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained pipeline"""
        self.model.load(filepath)
        self.pipeline = self.model.pipeline
        print(f"✅ Model loaded from {filepath}")