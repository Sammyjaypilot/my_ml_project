import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
)
import os
import yaml
from pathlib import Path
import json


def load_params(
    param_file_path="C:/Users/DELL/Desktop/my_ml_project/notebook/params.yaml",
):
    """Load parameters from params.yaml."""
    if not os.path.exists(param_file_path):
        raise FileNotFoundError(
            f"{param_file_path} not found. Please check your project root."
        )

    with open(param_file_path, "r") as f:
        params = yaml.safe_load(f)

    return params


def get_data_preprocessor(numerical_cols, categorical_cols):
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


class DataProcessor:
    """
    Handles data splitting, preprocessing, and feature engineering
    """

    def __init__(
        self,
        target_column="Fever_Severity_code",
        test_size=0.2,
        val_size=0.1,
        random_state=42,
    ):
        self.target_column = target_column
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.preprocessor = None
        self.feature_columns = None

    def split_data(self, df_fever, save_path=None, generate_metrics=True):
        """
        Split data into train, validation, and test sets
        """
        if self.target_column not in df_fever.columns:
            if (
                "Fever_Severity" in df_fever.columns
                and "Fever_Severity_code" not in df_fever.columns
            ):
                le = LabelEncoder()
                df_fever["Fever_Severity_code"] = le.fit_transform(
                    df_fever["Fever_Severity"]
                )
                print("✅ Converted 'Fever_Severity' → 'Fever_Severity_code'")
            elif "Fever_Severity_code" in df_fever.columns:
                print("ℹ️  'Fever_Severity_code' already exists, skipping encoding.")
            else:
                raise KeyError(
                    "Neither 'Fever_Severity' nor 'Fever_Severity_code' column found in dataset."
                )

        X = df_fever.drop(columns=[self.target_column])
        y = df_fever[self.target_column]

        # Separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Separate validation set from temporary set
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=y_temp,
        )

        if save_path:
            self._save_splits(X_train, X_val, X_test, y_train, y_val, y_test, save_path)

        if generate_metrics:
            self._generate_data_metrics(
                df_fever, X_train, X_val, X_test, y_train, y_val, y_test
            )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _save_splits(
        self, X_train, X_val, X_test, y_train, y_val, y_test, save_path, verbose=True
    ):
        """
        Save train, validation, and test splits to CSV files
        """
        os.makedirs(save_path, exist_ok=True)

        # Combine features and targets for each split
        train_data = X_train.copy()
        train_data[self.target_column] = y_train

        val_data = X_val.copy()
        val_data[self.target_column] = y_val

        test_data = X_test.copy()
        test_data[self.target_column] = y_test

        train_data.to_csv(os.path.join(save_path, "train.csv"), index=False)
        val_data.to_csv(os.path.join(save_path, "val.csv"), index=False)
        test_data.to_csv(os.path.join(save_path, "test.csv"), index=False)

        if verbose:
            print(f"✅ Data splits saved to {save_path}")
            print(f"   Train set: {len(train_data)} samples")
            print(f"   Validation set: {len(val_data)} samples")
            print(f"   Test set: {len(test_data)} samples")

    def _generate_data_metrics(
        self, df_fever, X_train, X_val, X_test, y_train, y_val, y_test
    ):
        """Generate comprehensive data metrics"""
        metrics = {
            "dataset": {
                "total_samples": len(df_fever),
                "total_features": df_fever.shape[1] - 1,
                "missing_values_total": df_fever.isnull().sum().sum(),
                "missing_values_percentage": (
                    df_fever.isnull().sum().sum()
                    / (df_fever.shape[0] * df_fever.shape[1])
                    * 100
                ),
            },
            "splits": {
                "train_samples": len(X_train),
                "validation_samples": len(X_val),
                "test_samples": len(X_test),
                "train_percentage": (len(X_train) / len(df_fever) * 100),
                "validation_percentage": (len(X_val) / len(df_fever) * 100),
                "test_percentage": (len(X_test) / len(df_fever) * 100),
            },
            "class_distribution": {
                "train": {
                    "class_0": int((y_train == 0).sum()),
                    "class_1": int((y_train == 1).sum()),
                    "class_0_percentage": float((y_train == 0).mean() * 100),
                    "class_1_percentage": float((y_train == 1).mean() * 100),
                },
                "validation": {
                    "class_0": int((y_val == 0).sum()),
                    "class_1": int((y_val == 1).sum()),
                    "class_0_percentage": float((y_val == 0).mean() * 100),
                    "class_1_percentage": float((y_val == 1).mean() * 100),
                },
                "test": {
                    "class_0": int((y_test == 0).sum()),
                    "class_1": int((y_test == 1).sum()),
                    "class_0_percentage": float((y_test == 0).mean() * 100),
                    "class_1_percentage": float((y_test == 1).mean() * 100),
                },
            },
            "data_quality": {
                "is_balanced": abs((y_train == 1).mean() - 0.5) < 0.2,
                "has_missing_values": df_fever.isnull().sum().sum() > 0,
                "split_ratios_correct": abs(len(X_train) / len(df_fever) - 0.7) < 0.1,
            },
        }

        self._save_data_metrics(metrics)
        return metrics

    def _save_data_metrics(
        self,
        metrics,
        filepath="C:/Users/DELL/Desktop/my_ml_project/metrics/metrics.json",
    ):
        """Save data metrics to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"✅ Data metrics saved to {filepath}")

    def create_preprocessor(self):
        """
        Create preprocessing pipeline using parameters from params.yaml
        """
        # Load parameters
        params = load_params()

        # Get preprocessing parameters with defaults
        preprocessing_params = params.get("preprocessing", {})
        numerical_params = preprocessing_params.get("numerical", {})
        categorical_params = preprocessing_params.get("categorical", {})

        # Numerical transformer
        numerical_imputer = numerical_params.get("imputer", "median")
        numerical_scaler_type = numerical_params.get("scaler", "standard")

        if numerical_scaler_type == "standard":
            scaler = StandardScaler()
        elif numerical_scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif numerical_scaler_type == "robust":
            scaler = RobustScaler()
        else:
            print(f"⚠️  Unknown scaler: {numerical_scaler_type}. Using StandardScaler.")
            scaler = StandardScaler()

        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=numerical_imputer)),
                ("scaler", scaler),
            ]
        )

        # Categorical transformer
        categorical_imputer = categorical_params.get("imputer", "most_frequent")
        handle_unknown = categorical_params.get("handle_unknown", "ignore")

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=categorical_imputer)),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False),
                ),
            ]
        )

        # Get feature columns
        prepare_params = params.get("prepare", {})
        numerical_cols = prepare_params.get(
            "numerical_cols",
            ["Temperature", "Age", "BMI", "Humidity", "AQI", "Heart_Rate"],
        )
        categorical_cols = prepare_params.get(
            "categorical_cols",
            [
                "Gender",
                "Headache",
                "Body_Ache",
                "Fatigue",
                "Chronic_Conditions",
                "Allergies",
                "Smoking_History",
                "Alcohol_Consumption",
                "Physical_Activity",
                "Diet_Type",
                "Blood_Pressure",
                "Previous_Medication",
                "Recommended_Medication",
            ],
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        print(f"✅ Created preprocessor with:")
        print(
            f"   Numerical: {numerical_imputer} imputer, {numerical_scaler_type} scaler"
        )
        print(f"   Categorical: {categorical_imputer} imputer, onehot encoder")

        return self.preprocessor

    def get_feature_names(self):
        """Get feature names after preprocessing"""
        if self.preprocessor is None:
            return None
        return self.preprocessor.get_feature_names_out()


if __name__ == "__main__":
    # Load parameters
    params = load_params()

    # Get preprocess parameters (these are tracked by DVC)
    preprocess_params = params.get("preprocess", {})
    test_size = preprocess_params.get("test_size", 0.2)
    val_size = preprocess_params.get("val_size", 0.1)
    random_state = preprocess_params.get("random_state", 42)

    # Get other parameters
    prepare_params = params.get("prepare", {})
    raw_path = prepare_params.get(
        "rawdata_file_path", "C:/Users/DELL/Desktop/my_ml_project/notebook/fever.csv"
    )
    target_col = prepare_params.get("target_column", "Fever_Severity_code")

    print(f"Using DVC-tracked parameters:")
    print(f"  test_size: {test_size}")
    print(f"  val_size: {val_size}")
    print(f"  random_state: {random_state}")

    # Load data
    print(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)

    # Initialize DataProcessor with DVC-tracked parameters
    dp = DataProcessor(
        target_column=target_col,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    # Create preprocessor
    dp.create_preprocessor()

    # Process and save data
    save_dir = "C:/Users/DELL/Desktop/my_ml_project/data/processed"
    dp.split_data(df, save_path=save_dir, generate_metrics=True)

    print("✅ Preprocessing completed successfully!")
    print(f"Processed data saved to: {save_dir}")