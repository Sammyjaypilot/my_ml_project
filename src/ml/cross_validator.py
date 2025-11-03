# hyper-parameter tuning
"""
Cross Validation Module for Hypertension Prediction
Provides comprehensive cross-validation strategies and model evaluation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_validate, 
    StratifiedKFold, 
    KFold,
    cross_val_predict
)
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Any
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
    

class CrossValidator:
    """
    Comprehensive cross-validation and model evaluation class
    """
    
    def __init__(self, pipeline=None, cv_strategy='stratified', n_splits=5, 
                 random_state=42, scoring_metrics=None):
        """
        Initialize CrossValidator
        
        Args:
            pipeline: sklearn Pipeline or model object
            cv_strategy (str): 'stratified', 'kfold', or 'timeseries'
            n_splits (int): Number of cross-validation folds
            random_state (int): Random seed for reproducibility
            scoring_metrics (dict): Custom scoring metrics
        """
        self.pipeline = pipeline
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv_strategy = cv_strategy
        self.cv_results_ = None
        self.feature_importance_ = None
        
        # Default scoring metrics for classification
        if scoring_metrics is None:
            self.scoring_metrics = {
                'accuracy': 'accuracy',
                'precision': 'precision_weighted',
                'recall': 'recall_weighted', 
                'f1': 'f1_weighted',
                'roc_auc': 'roc_auc_ovr_weighted'
            }
        else:
            self.scoring_metrics = scoring_metrics
            
        # Initialize cross-validation strategy
        self.cv = self._get_cv_strategy()
    
    def _get_cv_strategy(self):
        """Get the cross-validation strategy object"""
        if self.cv_strategy == 'stratified':
            return StratifiedKFold(
                n_splits=self.n_splits, 
                shuffle=True, 
                random_state=self.random_state
            )
        elif self.cv_strategy == 'kfold':
            return KFold(
                n_splits=self.n_splits, 
                shuffle=True, 
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unsupported CV strategy: {self.cv_strategy}")
    
    def cross_validate(self, X, y, return_train_score=True, return_estimator=True):
        """
        Perform cross-validation on the given data
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target vector
            return_train_score (bool): Whether to return training scores
            return_estimator (bool): Whether to return fitted estimators
            
        Returns:
            dict: Cross-validation results
        """
        if self.pipeline is None:
            raise ValueError("Pipeline must be set before cross-validation")
        
        print(f"🔍 Performing {self.n_splits}-fold {self.cv_strategy} cross-validation...")
        
        # Perform cross-validation
        self.cv_results_ = cross_validate(
            estimator=self.pipeline,
            X=X,
            y=y,
            cv=self.cv,
            scoring=self.scoring_metrics,
            return_train_score=return_train_score,
            return_estimator=return_estimator,
            n_jobs=-1  # Use all available cores
        )
        
        self._print_cv_summary()
        return self.cv_results_
    
    def _print_cv_summary(self):
        """Print a summary of cross-validation results"""
        if self.cv_results_ is None:
            print("No cross-validation results available. Run cross_validate() first.")
            return
        
        print("\n📊 Cross-Validation Results Summary:")
        print("=" * 50)
        
        # Test scores
        for metric in self.scoring_metrics.keys():
            test_scores = self.cv_results_[f'test_{metric}']
            mean_test = np.mean(test_scores)
            std_test = np.std(test_scores)
            
            print(f"{metric.capitalize():<12}: {mean_test:.4f} (+/- {std_test:.4f})")
            
            # Print training scores if available
            if f'train_{metric}' in self.cv_results_:
                train_scores = self.cv_results_[f'train_{metric}']
                mean_train = np.mean(train_scores)
                print(f"  (Train)     : {mean_train:.4f}")
    
    def get_cv_predictions(self, X, y, method='predict_proba'):
        """
        Get cross-validated predictions
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target vector
            method (str): 'predict' or 'predict_proba'
            
        Returns:
            array: Cross-validated predictions
        """
        if self.pipeline is None:
            raise ValueError("Pipeline must be set before getting predictions")
        
        return cross_val_predict(
            estimator=self.pipeline,
            X=X,
            y=y,
            cv=self.cv,
            method=method,
            n_jobs=-1
        )
    
    def evaluate_model(self, X, y, save_plots=False, plot_path=None):
        """
        Comprehensive model evaluation with cross-validation
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target vector
            save_plots (bool): Whether to save evaluation plots
            plot_path (str): Path to save plots
            
        Returns:
            dict: Comprehensive evaluation metrics
        """
        print("🎯 Performing Comprehensive Model Evaluation...")
        
        # Get cross-validated predictions
        y_pred_proba = self.get_cv_predictions(X, y, method='predict_proba')
        y_pred = np.argmax(y_pred_proba, axis=1) if y_pred_proba.ndim > 1 else (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0),
        }
        
        # Add ROC-AUC if binary or multiclass
        try:
            if len(np.unique(y)) == 2:  # Binary classification
                metrics['roc_auc'] = roc_auc_score(y, y_pred_proba[:, 1])
            else:  # Multiclass classification
                metrics['roc_auc'] = roc_auc_score(y, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            metrics['roc_auc'] = None
        
        # Print metrics
        print("\n📈 Comprehensive Evaluation Metrics:")
        print("=" * 40)
        for metric, value in metrics.items():
            if value is not None:
                print(f"{metric.capitalize():<12}: {value:.4f}")
        
        # Generate and plot confusion matrix
        self._plot_confusion_matrix(y, y_pred, save_plots, plot_path)
        
        # Plot ROC curve if binary classification
        if len(np.unique(y)) == 2:
            self._plot_roc_curve(y, y_pred_proba[:, 1], save_plots, plot_path)
        
        return metrics
    
    def _plot_confusion_matrix(self, y_true, y_pred, save_plots=False, plot_path=None):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(y_true), 
                   yticklabels=np.unique(y_true))
        plt.title('Confusion Matrix - Cross Validation')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_plots and plot_path:
            os.makedirs(plot_path, exist_ok=True)
            plt.savefig(os.path.join(plot_path, 'confusion_matrix.png'), 
                       bbox_inches='tight', dpi=300)
        plt.show()
    
    def _plot_roc_curve(self, y_true, y_pred_proba, save_plots=False, plot_path=None):
        """Plot and save ROC curve (for binary classification)"""
        from sklearn.metrics import roc_curve
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_plots and plot_path:
            plt.savefig(os.path.join(plot_path, 'roc_curve.png'), 
                       bbox_inches='tight', dpi=300)
        plt.show()
    
    def calculate_feature_importance(self, X, feature_names=None):
        """
        Calculate feature importance from cross-validated models
        
        Args:
            X (array-like): Feature matrix for determining number of features
            feature_names (list): Names of features
            
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if self.cv_results_ is None or 'estimator' not in self.cv_results_:
            raise ValueError("Run cross_validate with return_estimator=True first")
        
        # Get estimators from cross-validation
        estimators = self.cv_results_['estimator']
        
        # Initialize feature importance storage
        n_features = X.shape[1] if hasattr(X, 'shape') else len(X.columns)
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        importance_scores = []
        
        for i, estimator in enumerate(estimators):
            # Extract the actual model from the pipeline
            if hasattr(estimator, 'named_steps'):
                model = estimator.named_steps['classifier']
            else:
                model = estimator
            
            # Get feature importance based on model type
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])  # For linear models
            else:
                print(f"⚠️  Model {i} doesn't have feature importance method")
                importance = np.zeros(n_features)
            
            importance_scores.append(importance)
        
        # Calculate mean importance across folds
        mean_importance = np.mean(importance_scores, axis=0)
        std_importance = np.std(importance_scores, axis=0)
        
        # Create feature importance DataFrame
        self.feature_importance_ = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_importance,
            'std': std_importance
        }).sort_values('importance', ascending=False)
        
        return self.feature_importance_
    
    def plot_feature_importance(self, top_k=15, save_plot=False, plot_path=None):
        """Plot feature importance"""
        if self.feature_importance_ is None:
            raise ValueError("Run calculate_feature_importance() first")
        
        # Select top K features
        top_features = self.feature_importance_.head(top_k)
        
        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(top_features))
        
        plt.barh(y_pos, top_features['importance'], xerr=top_features['std'], 
                align='center', alpha=0.7, capsize=5)
        plt.yticks(y_pos, top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_k} Most Important Features (Cross-Validated)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_plot and plot_path:
            os.makedirs(plot_path, exist_ok=True)
            plt.savefig(os.path.join(plot_path, 'feature_importance.png'), 
                       bbox_inches='tight', dpi=300)
        plt.show()
    
    def save_results(self, filepath):
        """Save cross-validation results to JSON file"""
        if self.cv_results_ is None:
            raise ValueError("No results to save. Run cross_validate() first.")
        
        # Convert numpy arrays to lists for JSON serialization
        results_to_save = {}
        for key, value in self.cv_results_.items():
            if hasattr(value, 'tolist'):
                results_to_save[key] = value.tolist()
            else:
                results_to_save[key] = value
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"✅ Cross-validation results saved to {filepath}")


class HyperparameterTuner:
    """
    Advanced hyperparameter tuning with cross-validation
    """
    
    def __init__(self, pipeline, param_grid, cv_strategy='stratified', n_splits=5):
        self.pipeline = pipeline
        self.param_grid = param_grid
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.best_params_ = None
        self.best_score_ = None
    
    def grid_search(self, X, y, scoring='accuracy', n_jobs=-1):
        """Perform grid search with cross-validation"""
        from sklearn.model_selection import GridSearchCV
        
        cv = CrossValidator(cv_strategy=self.cv_strategy, n_splits=self.n_splits).cv
        
        grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=self.param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            return_train_score=True
        )
        
        print("🔍 Performing Grid Search...")
        grid_search.fit(X, y)
        
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        
        print(f"✅ Best parameters: {self.best_params_}")
        print(f"✅ Best {scoring} score: {self.best_score_:.4f}")
        
        return grid_search



if __name__ == '__main__':
    # create a sample pipeline
    pipeline = Pipeline([
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Initialize cross-validator
    cv = CrossValidator(pipeline=pipeline, n_splits=5)
    
    print("CrossValidator module loaded successfully!")
    print("Usage:")
    print("1. Initialize with your pipeline")
    print("2. Run cross_validate(X, y)")
    print("3. Analyze results using the provided methods")