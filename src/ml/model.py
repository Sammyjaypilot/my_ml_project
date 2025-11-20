# src/ml/models.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
import joblib
import json
import yaml
from pathlib import Path
import mlflow
import mlflow.sklearn

# Import your existing data processor
try:
    from ml.data_processor import DataProcessor, load_params
except ImportError:
    from data_processor import DataProcessor, load_params

class ModelTrainer:
    """
    Professional model training and evaluation with pipeline integration
    """
    
    def __init__(self, params_path: Path = None):
        self.params_path = params_path or Path("C:/Users/DELL/Desktop/my_ml_project/notebook/params.yaml")
        self.params = load_params(self.params_path)
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def build_pipelines(self, preprocessor):
        """
        Build multiple model pipelines using parameters from params.yaml
        """
        model_params = self.params.get('model', {})
        
        pipelines = {}
        
        # Random Forest
        if 'random_forest' in model_params:
            rf_params = model_params['random_forest']
            pipelines['RandomForest'] = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=rf_params.get('n_estimators', 100),
                    max_depth=rf_params.get('max_depth', None),
                    min_samples_split=rf_params.get('min_samples_split', 2),
                    min_samples_leaf=rf_params.get('min_samples_leaf', 1),
                    random_state=model_params.get('random_state', 42),
                    n_jobs=-1
                ))
            ])
        
        # Logistic Regression
        if 'logistic_regression' in model_params:
            lr_params = model_params['logistic_regression']
            pipelines['LogisticRegression'] = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(
                    C=lr_params.get('C', 1.0),
                    penalty=lr_params.get('penalty', 'l2'),
                    solver=lr_params.get('solver', 'lbfgs'),
                    max_iter=lr_params.get('max_iter', 1000),
                    random_state=model_params.get('random_state', 42)
                ))
            ])
        
        # XGBoost
        if 'xgboost' in model_params:
            xgb_params = model_params['xgboost']
            pipelines['XGBoost'] = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', XGBClassifier(
                    n_estimators=xgb_params.get('n_estimators', 100),
                    max_depth=xgb_params.get('max_depth', 6),
                    learning_rate=xgb_params.get('learning_rate', 0.1),
                    subsample=xgb_params.get('subsample', 1.0),
                    colsample_bytree=xgb_params.get('colsample_bytree', 1.0),
                    random_state=model_params.get('random_state', 42),
                    n_jobs=-1
                ))
            ])
        
        # SVM
        if 'svm' in model_params:
            svm_params = model_params['svm']
            pipelines['SVM'] = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', SVC(
                    C=svm_params.get('C', 1.0),
                    kernel=svm_params.get('kernel', 'rbf'),
                    probability=True,
                    random_state=model_params.get('random_state', 42)
                ))
            ])
        
        # K-Nearest Neighbors
        if 'knn' in model_params:
            knn_params = model_params['knn']
            pipelines['KNN'] = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', KNeighborsClassifier(
                    n_neighbors=knn_params.get('n_neighbors', 5),
                    weights=knn_params.get('weights', 'uniform')
                ))
            ])
        
        # Decision Tree
        if 'decision_tree' in model_params:
            dt_params = model_params['decision_tree']
            pipelines['DecisionTree'] = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', DecisionTreeClassifier(
                    max_depth=dt_params.get('max_depth', None),
                    min_samples_split=dt_params.get('min_samples_split', 2),
                    min_samples_leaf=dt_params.get('min_samples_leaf', 1),
                    random_state=model_params.get('random_state', 42)
                ))
            ])
        
        print(f"✅ Built {len(pipelines)} model pipelines: {list(pipelines.keys())}")
        return pipelines
    
    def evaluate_model(self, pipeline, X_test, y_test, model_name):
        """
        Comprehensive model evaluation
        """
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Get probabilities for AUC (handle models that don't have predict_proba)
        try:
            y_pred_proba = pipeline.predict_proba(X_test)
            if y_pred_proba.shape[1] >= 2:
                y_proba_pos = y_pred_proba[:, 1]
                auc = roc_auc_score(y_test, y_proba_pos)
            else:
                auc = -1.0
        except (AttributeError, IndexError):
            auc = -1.0
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': report.get('weighted avg', {}).get('precision', -1.0),
            'recall': report.get('weighted avg', {}).get('recall', -1.0),
            'f1_score': f1,
            'roc_auc': auc
        }
        
        # Print results
        print(f"\n--- {model_name} ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(classification_report(y_test, y_pred))
        
        return metrics
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, preprocessor, use_mlflow=True):
        """
        Train and evaluate all models
        """
        # Build pipelines
        pipelines = self.build_pipelines(preprocessor)
        
        # MLflow setup
        mlflow_enabled = False
        if use_mlflow:
            try:
                tracking_uri = self.params.get('training', {}).get('tracking_uri')
                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                    mlflow_enabled = True
                    print("✅ MLflow tracking enabled")
            except Exception as e:
                print(f"⚠️ MLflow disabled: {e}")
        
        # Train and evaluate each model
        for model_name, pipeline in pipelines.items():
            print(f"\n🚀 Training {model_name}...")
            
            if mlflow_enabled:
                try:
                    experiment_name = self.params.get('training', {}).get('experiment_name', 'FeverSeverity_Comparison')
                    mlflow.set_experiment(experiment_name)
                    
                    with mlflow.start_run(run_name=f"{model_name}_baseline"):
                        # Log parameters
                        mlflow.log_param("model_type", model_name)
                        mlflow.log_param("random_state", self.params.get('model', {}).get('random_state', 42))
                        
                        # Train model
                        pipeline.fit(X_train, y_train)
                        
                        # Evaluate
                        metrics = self.evaluate_model(pipeline, X_test, y_test, model_name)
                        
                        # Log metrics
                        mlflow.log_metrics(metrics)
                        
                        # Log model
                        mlflow.sklearn.log_model(pipeline, "model")
                        
                        # Store results
                        self.results[model_name] = {
                            'pipeline': pipeline,
                            'metrics': metrics
                        }
                        
                        print(f"✅ {model_name} trained and logged to MLflow")
                        
                except Exception as e:
                    print(f"⚠️ MLflow failed for {model_name}: {e}")
                    # Fallback without MLflow
                    pipeline.fit(X_train, y_train)
                    metrics = self.evaluate_model(pipeline, X_test, y_test, model_name)
                    self.results[model_name] = {
                        'pipeline': pipeline,
                        'metrics': metrics
                    }
            else:
                # Train without MLflow
                pipeline.fit(X_train, y_train)
                metrics = self.evaluate_model(pipeline, X_test, y_test, model_name)
                self.results[model_name] = {
                    'pipeline': pipeline,
                    'metrics': metrics
                }
        
        return self.results
    
    def select_best_model(self, metric='roc_auc'):
        """
        Select the best model based on specified metric
        """
        if not self.results:
            raise ValueError("No results available. Run train_and_evaluate first.")
        
        # Filter models that have the metric
        valid_models = {name: result for name, result in self.results.items() 
                       if result['metrics'].get(metric, -1) > 0}
        
        if not valid_models:
            # Fallback to accuracy if ROC-AUC not available
            metric = 'accuracy'
            valid_models = self.results
        
        self.best_model_name = max(valid_models, 
                                 key=lambda x: valid_models[x]['metrics'].get(metric, -1))
        self.best_model = self.results[self.best_model_name]['pipeline']
        
        print(f"\n🎉 Best Model: {self.best_model_name}")
        print(f"🏆 Best {metric.upper()}: {self.results[self.best_model_name]['metrics'].get(metric, -1):.4f}")
        
        return self.best_model_name, self.best_model
    
    def save_best_model(self, model_path: Path):
        """
        Save the best model
        """
        if self.best_model is None:
            raise ValueError("No best model selected. Run select_best_model first.")
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model, model_path)
        print(f"✅ Best model saved to: {model_path}")
        
        # Save comparison results
        results_path = model_path.parent / "model_comparison.json"
        comparison_data = {
            'best_model': self.best_model_name,
            'comparison_date': pd.Timestamp.now().isoformat(),
            'models': {
                name: {
                    'metrics': result['metrics'],
                    'parameters': str(result['pipeline'].get_params())
                } for name, result in self.results.items()
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"✅ Model comparison saved to: {results_path}")

def main():
    """
    Main function to run the complete model training pipeline
    """
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load processed data using your existing DataProcessor
    data_processor = DataProcessor()
    X_train, X_test, y_train, y_test = data_processor.load_processed_data()
    
    # Build preprocessor using your existing method
    preprocessor = data_processor.create_preprocessor()
    
    # Train and evaluate all models
    print("🔬 Starting model comparison...")
    results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)
    
    # Select best model
    best_name, best_model = trainer.select_best_model(metric='roc_auc')
    
    # Save best model
    model_path = Path("C:/Users/DELL/Desktop/my_ml_project/models/best_fever_model.pkl")
    trainer.save_best_model(model_path)
    
    print("\n🎉 Model training pipeline completed!")
    return trainer

if __name__ == "__main__":
    trainer = main()

    