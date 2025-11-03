import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectPercentile, chi2
import os
import yaml
from pathlib import Path



def load_params(params_path="params.yaml"):
    """Load parameters from params.yaml."""
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"{params_path} not found. Please check your project root.")
    
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    
    return params

def get_data_preprocessor(numerical_cols, categorical_cols):
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
        #('selector', SelectPercentile(chi2, percentile=50))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    return preprocessor


class DataProcessor:
    """
    Handles data splitting, preprocessing, and feature engineering
    """

    def __init__(self, target_column='Fever_Severity_code', test_size=0.2, val_size=0.1, random_state=42):
        self.target_column = target_column
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.preprocessor = None
        self.feature_columns = None

    def split_data(self, df_fever, save_path=None, generate_metrics=True):
        """
        Split data into train, validation, and test sets

        Args:
            data (pd.DataFrame): Raw data
            save_path (str): Path to save processed data (optional)
            generate_metrics (bool): Whether to generate data metrics

        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        X = df_fever.drop(columns=[self.target_column])
        y = df_fever[self.target_column]

        # separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # separate validation set from temporary set
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state, stratify=y_temp
        )

        if save_path:
            self._save_splits(X_train, X_val, X_test,
                              y_train, y_val, y_test, save_path)

        if generate_metrics:
            self._generate_data_metrics(
                df_fever, X_train, X_val, X_test, y_train, y_val, y_test, save_path)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _save_splits(self, X_train, X_val, X_test, y_train, y_val, y_test, save_path, verbose=False):
        """
        Save train, validation, and test splits to CSV files
        """
        # create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Combine features and targets for each split
        train_data = X_train.copy()
        train_data[self.target_column] = y_train

        val_data = X_val.copy()
        val_data[self.target_column] = y_val

        test_data = X_test.copy()
        test_data[self.target_column] = y_test

        train_data.to_csv(os.path.join(save_path, 'train.csv'), index=False)
        val_data.to_csv(os.path.join(save_path, 'val.csv'), index=False)
        test_data.to_csv(os.path.join(save_path, 'test.csv'), index=False)

        if verbose:
            print(f"✅ Data splits saved to {save_path}")
            print(f"   Train set: {len(train_data)} samples")
            print(f"   Validation set: {len(val_data)} samples")
            print(f"   Test set: {len(test_data)} samples")

    def _generate_data_metrics(self, df_fever, X_train, X_val, X_test, y_train, y_val, y_test, save_metrics=True):
        """Generate comprehensive data metrics"""

        metrics = {
            "dataset": {
                "total_samples": len(df_fever),
                "total_features": df_fever.shape[1] - 1,  # excluding target
                "missing_values_total": df_fever.isnull().sum().sum(),
                "missing_values_percentage": (df_fever.isnull().sum().sum() / (df_fever.shape[0] * df_fever.shape[1]) * 100)
            },
            "splits": {
                "train_samples": len(X_train),
                "validation_samples": len(X_val),
                "test_samples": len(X_test),
                "train_percentage": (len(X_train) / len(df_fever) * 100),
                "validation_percentage": (len(X_val) / len(df_fever) * 100),
                "test_percentage": (len(X_test) / len(df_fever) * 100)
            },
            "class_distribution": {
                "train": {
                    "class_0": int((y_train == 0).sum()),
                    "class_1": int((y_train == 1).sum()),
                    "class_0_percentage": float((y_train == 0).mean() * 100),
                    "class_1_percentage": float((y_train == 1).mean() * 100)
                },
                "validation": {
                    "class_0": int((y_val == 0).sum()),
                    "class_1": int((y_val == 1).sum()),
                    "class_0_percentage": float((y_val == 0).mean() * 100),
                    "class_1_percentage": float((y_val == 1).mean() * 100)
                },
                "test": {
                    "class_0": int((y_test == 0).sum()),
                    "class_1": int((y_test == 1).sum()),
                    "class_0_percentage": float((y_test == 0).mean() * 100),
                    "class_1_percentage": float((y_test == 1).mean() * 100)
                }
            },
            "feature_statistics": {
                "numerical_features": ['Temperature','Age', 'BMI', 'Humidity', 'AQI', 'Heart_Rate'],
                "categorical_features": ['Gender', 'Headache', 'Body_Ache', 'Fatigue','Chronic_Conditions', 'Allergies', 'Smoking_History','Alcohol_Consumption', 'Physical_Activity', 'Diet_Type','Blood_Pressure', 'Previous_Medication', 'Recommended_Medication'],
                "target_variable": "Fever_Severity_code"
            },
            "data_quality": {
                # Within 20% of balanced
                "is_balanced": abs((y_train == 1).mean() - 0.5) < 0.2,
                "has_missing_values": df_fever.isnull().sum().sum() > 0,
                # ~70% train
                "split_ratios_correct": abs(len(X_train)/len(df_fever) - 0.7) < 0.1
            }
        }
        if save_metrics:
            self._save_data_metrics(
                metrics, filepath="reports/data_metrics.json")

        return metrics

    def _save_data_metrics(self, metrics, filepath="reports/data_metrics.json"):
        """Save data metrics to JSON file"""
        import json
        import os

        # Create reports directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        print(f"✅ Data metrics saved to {filepath}")

    def create_preprocessor(self, numerical_cols=None, categorical_cols=None, 
                           numerical_imputer=None, numerical_scaler=None,
                           categorical_imputer=None, categorical_encoder=None,
                           handle_unknown=None):
        """
        Create preprocessing pipeline using parameters from params.yaml
        """
        current_file = Path(__file__)
        params_file_path = current_file.parent.parent.parent / 'params.yaml'
        print(f"DEBUG: Attempting to open file at: {params_file_path}") # Verify path one last time

        # Load default parameters from params.yaml if not provided
        try:
            with open(params_file_path, 'r') as f:
                params = yaml.safe_load(f)
            preprocessing_params = params['preprocessing']
            
            numerical_imputer = numerical_imputer or preprocessing_params['numerical']['imputer']
            numerical_scaler = numerical_scaler or preprocessing_params['numerical']['scaler']
            categorical_imputer = categorical_imputer or preprocessing_params['categorical']['imputer']
            categorical_encoder = categorical_encoder or preprocessing_params['categorical']['encoder']
            handle_unknown = handle_unknown or preprocessing_params['categorical']['handle_unknown']
            numerical_cols = params.get('prepare', {}).get('numerical_cols', 
                   ['Temperature', 'Age', 'BMI', 'Humidity', 'AQI', 'Heart_Rate'])
            categorical_cols = params.get('prepare', {}).get('categorical_cols',
                    ['Gender', 'Headache', 'Body_Ache', 'Fatigue','Chronic_Conditions', 'Allergies', 'Smoking_History','Alcohol_Consumption', 'Physical_Activity', 'Diet_Type','Blood_Pressure', 'Previous_Medication', 'Recommended_Medication'])
        except Exception as e:
            print(f"⚠️  Could not load params.yaml: {e}. Using defaults.")
            numerical_imputer = numerical_imputer or 'median'
            numerical_scaler = numerical_scaler or 'standard'
            categorical_imputer = categorical_imputer or 'most_frequent'
            categorical_encoder = categorical_encoder or 'onehot'
            handle_unknown = handle_unknown or 'ignore'
        
        # Create numerical transformer based on parameters
        if numerical_scaler == 'standard':
            scaler = StandardScaler()
        elif numerical_scaler == 'minmax':
            scaler = MinMaxScaler()
        elif numerical_scaler == 'robust':
            scaler = RobustScaler()
        else:
            print(f"⚠️  Unknown scaler: {numerical_scaler}. Using StandardScaler.")
            scaler = StandardScaler()
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=numerical_imputer)),
            ('scaler', scaler)
        ])
        
        # Create categorical transformer
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=categorical_imputer)),
            ('encoder', OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
        print(f"✅ Created preprocessor with:")
        print(f"   Numerical: {numerical_imputer} imputer, {numerical_scaler} scaler")
        print(f"   Categorical: {categorical_imputer} imputer, {categorical_encoder} encoder")
        
        return self.preprocessor

    def get_feature_names(self):
        """Get feature names after preprocessing"""
        if self.preprocessor is None:
            return None

        feature_names = []
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat':
                # Get one-hot encoded feature names
                encoded_features = transformer.named_steps['onehot'].get_feature_names_out(
                    columns)
                feature_names.extend(encoded_features)

        return feature_names