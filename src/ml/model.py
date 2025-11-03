# src/ml/model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')
import yaml

XGBOOST_AVAILABLE = False

class Model:
    def __init__(self, model_type='random_forest', random_state=42, **model_params):
        """
        Initialize model with specified type and parameters
        
        Args:
            model_type (str): Type of model to use
            random_state (int): Random seed for reproducibility
            **model_params: Additional model-specific parameters
        """
        self.pipeline = None
        self.model_type = model_type
        self.random_state = random_state
        self.model_params = model_params
        self.classifier = None
        self.initialise_model()
    
    def initialise_model(self):
        """Initialize the classifier based on model type using parameters from params.yaml"""
        try:
            with open('params.yaml', 'r') as f:
                params = yaml.safe_load(f)
            model_params = params['model']
        except:
            # Fallback to default parameters if params.yaml not available
            model_params = {}
        
        if self.model_type == 'random_forest':
            rf_params = model_params.get('random_forest', {})
            self.classifier = RandomForestClassifier(
                n_estimators=rf_params.get('n_estimators', self.model_params.get('n_estimators', 100)),
                max_depth=rf_params.get('max_depth', self.model_params.get('max_depth', None)),
                min_samples_split=rf_params.get('min_samples_split', self.model_params.get('min_samples_split', 2)),
                min_samples_leaf=rf_params.get('min_samples_leaf', self.model_params.get('min_samples_leaf', 1)),
                random_state=self.random_state,
                n_jobs=-1
            )
            
        elif self.model_type == 'logistic_regression':
            lr_params = model_params.get('logistic_regression', {})
            self.classifier = LogisticRegression(
                C=lr_params.get('C', self.model_params.get('C', 1.0)),
                penalty=lr_params.get('penalty', self.model_params.get('penalty', 'l2')),
                solver=lr_params.get('solver', self.model_params.get('solver', 'lbfgs')),
                max_iter=lr_params.get('max_iter', self.model_params.get('max_iter', 1000)),
                random_state=self.random_state,
                n_jobs=-1
            )
            
        elif self.model_type == 'xgboost':
            xgb_params = model_params.get('xgboost', {})
            self.classifier = XGBClassifier(
                n_estimators=xgb_params.get('n_estimators', self.model_params.get('n_estimators', 100)),
                max_depth=xgb_params.get('max_depth', self.model_params.get('max_depth', 6)),
                learning_rate=xgb_params.get('learning_rate', self.model_params.get('learning_rate', 0.1)),
                subsample=xgb_params.get('subsample', self.model_params.get('subsample', 1.0)),
                colsample_bytree=xgb_params.get('colsample_bytree', self.model_params.get('colsample_bytree', 1.0)),
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
            
        elif self.model_type == 'svm':
            svm_params = model_params.get('svm', {})
            self.classifier = SVC(
                C=svm_params.get('C', self.model_params.get('C', 1.0)),
                kernel=svm_params.get('kernel', self.model_params.get('kernel', 'rbf')),
                probability=True,
                random_state=self.random_state
            )
            
        elif self.model_type == 'knn':
            knn_params = model_params.get('knn', {})
            self.classifier = KNeighborsClassifier(
                n_neighbors=knn_params.get('n_neighbors', self.model_params.get('n_neighbors', 5)),
                weights=knn_params.get('weights', self.model_params.get('weights', 'uniform')),
                n_jobs=-1
            )
            
        elif self.model_type == 'decision_tree':
            dt_params = model_params.get('decision_tree', {})
            self.classifier = DecisionTreeClassifier(
                max_depth=dt_params.get('max_depth', self.model_params.get('max_depth', None)),
                min_samples_split=dt_params.get('min_samples_split', self.model_params.get('min_samples_split', 2)),
                min_samples_leaf=dt_params.get('min_samples_leaf', self.model_params.get('min_samples_leaf', 1)),
                random_state=self.random_state
            )
            
        else:
            available_models = ['random_forest', 'logistic_regression', 'xgboost', 'svm', 'knn', 'decision_tree']
            raise ValueError(f"Model type '{self.model_type}' not supported. Available models: {available_models}")
        
        print(f"✅ Initialized {self.model_type} classifier with parameters from params.yaml")
        
    def get_model_info(self):
        """Get information about the current model"""
        if self.classifier is None:
            return {"status": "Model not initialized"}
        
        info = {
            "model_type": self.model_type,
            "classifier": str(self.classifier.__class__.__name__),
            "parameters": self.classifier.get_params(),
            "is_fitted": hasattr(self.classifier, 'classes_')
        }
        return info
    
    def fit(self, X, y):
        """Fit the model to training data"""
        if self.pipeline is not None:
            self.pipeline.fit(X, y)
        elif self.classifier is not None:
            self.classifier.fit(X, y)
        else:
            raise ValueError("No model initialized. Call initialise_model() first.")
    
    def predict(self, X):
        """Make predictions"""
        if self.pipeline is not None:
            return self.pipeline.predict(X)
        elif self.classifier is not None:
            return self.classifier.predict(X)
        else:
            raise ValueError("No model available for prediction")
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.pipeline is not None:
            return self.pipeline.predict_proba(X)
        elif self.classifier is not None:
            # Check if classifier supports predict_proba
            if hasattr(self.classifier, 'predict_proba'):
                return self.classifier.predict_proba(X)
            else:
                raise ValueError(f"{self.model_type} does not support probability predictions")
        else:
            raise ValueError("No model available for prediction")
    
    def score(self, X, y):
        """Return the accuracy score"""
        if self.pipeline is not None:
            return self.pipeline.score(X, y)
        elif self.classifier is not None:
            return self.classifier.score(X, y)
        else:
            raise ValueError("No model available for scoring")
    
    def save(self, filepath):
        """Save the model/pipeline to file"""
        if self.pipeline is not None:
            joblib.dump(self.pipeline, filepath)
        elif self.classifier is not None:
            joblib.dump(self.classifier, filepath)
        else:
            raise ValueError("No model to save")
    
    def load(self, filepath):
        """Load a model/pipeline from file"""
        loaded_obj = joblib.load(filepath)
        
        # Determine if it's a pipeline or classifier
        if hasattr(loaded_obj, 'steps'):  # It's a pipeline
            self.pipeline = loaded_obj
            # Extract classifier from pipeline
            for name, step in loaded_obj.steps:
                if name == 'classifier':
                    self.classifier = step
                    break
        else:  # It's a classifier
            self.classifier = loaded_obj
        
        # Infer model type from the loaded object
        self._infer_model_type(loaded_obj)
    
    def _infer_model_type(self, model_obj):
        """Infer model type from loaded object"""
        model_class = model_obj.__class__.__name__
        
        if 'RandomForest' in model_class:
            self.model_type = 'random_forest'
        elif 'LogisticRegression' in model_class:
            self.model_type = 'logistic_regression'
        elif 'XGB' in model_class:
            self.model_type = 'xgboost'
        elif 'SVC' in model_class:
            self.model_type = 'svm'
        elif 'KNeighbors' in model_class:
            self.model_type = 'knn'
        elif 'DecisionTree' in model_class:
            self.model_type = 'decision_tree'
        else:
            self.model_type = 'unknown'