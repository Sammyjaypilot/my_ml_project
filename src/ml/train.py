import os
import json 
from pathlib import Path
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
import dvc.api
import dvc.repo
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import project's preprocessor builder
try:
    from ml.data_processor import get_data_preprocessor as project_get_preprocessor
    HAVE_PROJECT_PREPROCESSOR = True
except Exception:
    HAVE_PROJECT_PREPROCESSOR = False

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS_PATH = REPO_ROOT / "notebook" / "params.yaml"
TRAIN_CSV = REPO_ROOT / "data" / "processed" / "train.csv"
TEST_CSV = REPO_ROOT / "data" / "processed" / "test.csv"
METRICS_PATH = REPO_ROOT / "metrics" / "metrics.json"
MODEL_PATH = REPO_ROOT / "models" / "fever_severity_model.pkl"

def load_params(params_path: Path = DEFAULT_PARAMS_PATH):
    if not params_path.exists():
        print(f"⚠️ params.yaml not found at {params_path}. Using minimal default")
        return {}
    
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

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

def get_feature_columns():
    """Explicitly define which columns are features (EXCLUDING target)"""
    return [
        'Temperature', 'Age', 'Gender', 'BMI', 'Headache', 'Body_Ache',
        'Fatigue', 'Chronic_Conditions', 'Allergies', 'Smoking_History',
        'Alcohol_Consumption', 'Humidity', 'AQI', 'Physical_Activity',
        'Diet_Type', 'Heart_Rate', 'Blood_Pressure', 'Previous_Medication',
        'Recommended_Medication'
    ]  # 19 features - NO Fever_Severity!

def load_processed_data(train_path: Path = TRAIN_CSV, test_path: Path = TEST_CSV, target_col: str = "Fever_Severity_code"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Check if we need to convert Fever_Severity to numeric
    if target_col not in train.columns and 'Fever_Severity' in train.columns:
        print("⚠️ Converting 'Fever_Severity' to numeric 'Fever_Severity_code'")
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()

        # Convert training data
        train[target_col] = le.fit_transform(train['Fever_Severity'])

        # Convert test data to numeric
        test[target_col] = le.transform(test['Fever_Severity'])

        print(f"✅ Converted 'Fever_Severity' → '{target_col}' with classes: {list(le.classes_)}")

    # Get feature column explicitly
    feature_columns = get_feature_columns()

    if target_col not in train.columns or target_col not in test.columns:
        raise KeyError(f"Target column '{target_col}' not found in train/test CSVs. Available columns: {list(train.columns)}")
    
    X_train = train[feature_columns]
    y_train = train[target_col]
    X_test = test[feature_columns]
    y_test = test[target_col]

    print(f"✅ Loaded data: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
    print(f"✅ Target distribution - Train: {y_train.value_counts().to_dict()}, Test: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test

def build_default_preprocessor(numerical_cols, categorical_cols, preprocessing_params=None):
    preprocessing_params = preprocessing_params or {}
    num_imputer = SimpleImputer(strategy=preprocessing_params.get("numerical", {}).get("imputer", "median"))
    scaler = StandardScaler()
    num_pipe = Pipeline([("imputer", num_imputer), ("scaler", scaler)])

    cat_imputer = SimpleImputer(strategy=preprocessing_params.get("categorical", {}).get("imputer", "most_frequent"))
    
    handle_unknown = preprocessing_params.get("categorical", {}).get("handle_unknown", "ignore")
    
    # Use minimal parameters that work across all sklearn versions
    cat_encoder = OneHotEncoder(handle_unknown=handle_unknown)
    
    cat_pipe = Pipeline([("imputer", cat_imputer), ("onehot", cat_encoder)])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numerical_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor

def get_dvc_data_version_and_hash(repo_root: Path, rel_path: str):
    """Return (url, hash) for a given data file tracked by DVC."""
    url = None
    data_hash = None
    
    try:
        url = dvc.api.get_url(path=rel_path, repo=str(repo_root))
        print(f"✅ DVC data URL: {url}")
    except Exception as e:
        print(f"⚠️ Could not get DVC URL: {e}")
    
    # Return defaults - MLflow will still work without these
    return url, data_hash

def train_single_model(model_type, model_class, preprocessor, X_train, y_train, X_test, y_test, 
                      model_params, random_state, experiment_name, dvc_url=None, data_hash=None):
    """Train and evaluate a single model with MLflow tracking"""
    
    run_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        print(f"🏃 Training {model_type}...")
        
        # Log data provenance
        mlflow.log_param("dvc_repo", "https://github.com/Sammyjaypilot/my_ml_project")
        if dvc_url:
            mlflow.log_param("data_version_url", dvc_url)
        if data_hash:
            mlflow.log_param("data_hash", data_hash)

        # Log model parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("random_state", random_state)
        
        # Filter and log model-specific parameters
        valid_params = filter_valid_params(model_class, model_params)
        mlflow.log_params(valid_params)
        
        # Create classifier with filtered parameters
        classifier = model_class(**valid_params)
        if hasattr(classifier, 'random_state'):
            classifier.random_state = random_state
            
        # Create pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor), 
            ("classifier", classifier)
        ])
        
        # Train model
        start_time = datetime.now()
        pipeline.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        auc = -1.0
        if y_proba is not None and len(set(y_test)) >= 2:
            try:
                if y_proba.shape[1] == 2:
                    auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
            except Exception as e:
                logger.warning(f"Could not calculate AUC for {model_type}: {e}")
        
        # Log metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": auc,
            "training_time_seconds": training_time
        }
        
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(pipeline, "model")
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"✅ {model_type} completed in {training_time:.2f}s")
        print(f"   Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        return {
            "model_type": model_type,
            "pipeline": pipeline,
            "metrics": metrics,
            "classification_report": report,
            "run_id": mlflow.active_run().info.run_id
        }

def train_all_models():
    """Train all models defined in the configuration"""
    params = load_params()
    
    # Get parameters
    train_params = params.get("train", {})
    random_state = train_params.get("random_state", 42)
    model_config = params.get("model", {})
    
    print(f"Using DVC-tracked parameters:")
    print(f"  random_state: {random_state}")
    
    # Set up MLflow
    tracking_uri = params.get("training", {}).get("tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print("MLflow tracking URI set to:", tracking_uri)
    
    experiment_name = "FeverSeverity_Prediction"
    mlflow.set_experiment(experiment_name)
    print(f"🔬 Experiment: {experiment_name}")
    
    # DVC info
    rel_train_path = "data/processed/train.csv"
    dvc_url, data_hash = get_dvc_data_version_and_hash(REPO_ROOT, rel_train_path)
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Build preprocessor
    numerical_cols = params.get("prepare", {}).get("numerical_cols",
                                                     ['Temperature', 'Age', 'BMI', 'Humidity', 'AQI', 'Heart_Rate'])
    categorical_cols = params.get("prepare", {}).get("categorical_cols", [
        'Gender', 'Headache', 'Body_Ache', 'Fatigue', 'Chronic_Conditions', 'Allergies',
        'Smoking_History', 'Alcohol_Consumption', 'Physical_Activity', 'Diet_Type',
        'Blood_Pressure', 'Previous_Medication', 'Recommended_Medication'
    ])

    preproc_params = params.get("preprocessing", {})

    if HAVE_PROJECT_PREPROCESSOR:
        try:
            preprocessor = project_get_preprocessor(numerical_cols=numerical_cols, categorical_cols=categorical_cols)
            print("✅ Using project's get_data_preprocessor()")
        except Exception as e:
            print("⚠️ project_get_preprocessor failed, falling back to default preprocessor:", e)
            preprocessor = build_default_preprocessor(numerical_cols, categorical_cols, preproc_params)
    else:
        preprocessor = build_default_preprocessor(numerical_cols, categorical_cols, preproc_params)
    
    # Define all models to train
    MODELS_TO_TRAIN = {
        "random_forest": (RandomForestClassifier, model_config.get("random_forest", {})),
        "logistic_regression": (LogisticRegression, model_config.get("logistic_regression", {})),
        "xgboost": (XGBClassifier, model_config.get("xgboost", {})),
        "svm": (SVC, model_config.get("svm", {})),
        "knn": (KNeighborsClassifier, model_config.get("knn", {})),
        "decision_tree": (DecisionTreeClassifier, model_config.get("decision_tree", {}))
    }
    
    results = {}
    successful_models = []
    
    print(f"\n🎯 Training {len(MODELS_TO_TRAIN)} models...")
    print("=" * 60)
    
    for model_type, (model_class, model_params) in MODELS_TO_TRAIN.items():
        try:
            print(f"\n📊 Training {model_type}...")
            
            result = train_single_model(
                model_type=model_type,
                model_class=model_class,
                preprocessor=preprocessor,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model_params=model_params,
                random_state=random_state,
                experiment_name=experiment_name,
                dvc_url=dvc_url,
                data_hash=data_hash
            )
            
            results[model_type] = result
            successful_models.append(model_type)
            print(f"✅ {model_type} trained successfully!")
            
        except Exception as e:
            print(f"❌ {model_type} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save best model and generate summary
    if successful_models:
        # Find best model by F1 score
        best_model = None
        best_f1 = -1
        
        for model_type in successful_models:
            f1_score = results[model_type]["metrics"]["f1_score"]
            if f1_score > best_f1:
                best_f1 = f1_score
                best_model = model_type
        
        print(f"\n🎉 TRAINING SUMMARY")
        print("=" * 50)
        print(f"✅ Successful: {len(successful_models)}/{len(MODELS_TO_TRAIN)} models")
        print(f"🏆 Best Model: {best_model} (F1: {best_f1:.4f})")
        
        # Save best model for DVC tracking
        best_pipeline = results[best_model]["pipeline"]
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(best_pipeline, MODEL_PATH)
        print(f"💾 Best model saved: {MODEL_PATH}")
        
        # Save comprehensive metrics
        all_metrics = {model: result["metrics"] for model, result in results.items() if model in successful_models}
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(METRICS_PATH, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"📊 All metrics saved: {METRICS_PATH}")
        
        print(f"\n📈 View results at MLflow UI: {tracking_uri or 'http://localhost:5000'}")
        print(f"   Experiment: '{experiment_name}'")
    
    return results

if __name__ == "__main__":
    train_all_models()
