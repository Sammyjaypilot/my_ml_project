# experiment_tracker.py (FIXED IMPORTS)
import mlflow
import pandas as pd
from pathlib import Path
import sys
import os

# Add the src directory to Python path
project_root = Path("C:/Users/DELL/Desktop/my_ml_project")
src_path = project_root / "src"
sys.path.append(str(src_path))

# Now import from your modules
try:
    from training_pipeline import TrainingPipeline, filter_valid_params
    from data_processor import DataProcessor, load_params
    print("✅ Successfully imported from src modules")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Trying alternative import...")
    # Alternative import path
    sys.path.append(str(project_root))
    from .training_pipeline import TrainingPipeline, filter_valid_params
    from .data_processor import DataProcessor, load_params

# Define MODEL_CLASS_MAP
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

MODEL_CLASS_MAP = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "xgboost": XGBClassifier,
    "svm": SVC,
    "knn": KNeighborsClassifier,
    "decision_tree": DecisionTreeClassifier
}

def run_experiments():
    """Run comprehensive experiments"""
    
    # 1. Load parameters and get experiment name from CORRECT location
    params = load_params()
    experiment_name = params.get('training', {}).get('experiment_name', 'FeverSeverity_Prediction')
    model_config = params.get('model', {})
    
    print(f"🔬 Using experiment name: '{experiment_name}'")
    
    # 2. Set up MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)
    
    # 3. Load data
    print("📊 Loading data...")
    data_processor = DataProcessor()
    
    try:
        X_train, X_test, y_train, y_test = data_processor.load_processed_data()
        print(f"✅ Loaded processed data: {X_train.shape[0]} train, {X_test.shape[0]} test")
    except Exception as e:
        print(f"⚠️ Loading processed data failed: {e}")
        repo_root = Path("C:/Users/DELL/Desktop/my_ml_project")
        raw_path = repo_root / "notebook" / "fever.csv"
        df = pd.read_csv(raw_path)
        X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(
            df, save_path=str(repo_root / "data" / "processed"), generate_metrics=False
        )
        print(f"✅ Loaded and split raw data: {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    # 4. Create preprocessor
    print("🔧 Creating preprocessor...")
    preprocessor = data_processor.create_preprocessor()
    
    # 5. Train all models
    model_types = ["random_forest", "logistic_regression", "xgboost", "svm", "knn", "decision_tree"]
    successful_models = []
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"🏃 Training {model_type}...")
        print(f"{'='*50}")
        try:
            model_params = model_config.get(model_type, {})
            filtered_params = filter_valid_params(MODEL_CLASS_MAP[model_type], model_params)
            
            print(f"📋 Using parameters: {list(filtered_params.keys())}")
            
            tp = TrainingPipeline(
                preprocessor=preprocessor, 
                model_type=model_type, 
                model_params=filtered_params,
                random_state=model_config.get('random_state', 42)
            )
            
            # This will use the correct experiment name
            tp.fit(X_train, y_train, experiment_name=experiment_name)
            
            # Evaluate
            test_report = tp.get_comprehensive_report(X_train, y_train, X_test, y_test)
            tp.save_training_report(test_report, filepath=f"reports/{model_type}_metrics.json")
            
            # Register model
            tp.register_model(model_name="FeverSeverityModel", transition_to_stage="Staging")
            
            successful_models.append(model_type)
            print(f"✅ {model_type} completed successfully!")
            
        except Exception as e:
            print(f"❌ {model_type} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n🎉 EXPERIMENT SUMMARY")
    print(f"✅ Successful: {len(successful_models)}/{len(model_types)} models")
    print(f"📊 View results at: http://localhost:5000")
    print(f"   Look for experiment: '{experiment_name}'")

if __name__ == "__main__":
    run_experiments()