# src/Components/data_transformation.py

import os
import sys
import pickle
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.exception import CustomException
from src.logger import logging


class DataTransformation:
    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def get_data_transformer_object(self, X):
        """
        Creates a preprocessing pipeline that handles both numeric and categorical features.
        """
        try:
            categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
            numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

            logging.info(f"Detected numerical cols: {numerical_columns}")
            logging.info(f"Detected categorical cols: {categorical_columns}")

            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numerical_columns),
                ("cat", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Splitting features and target")

            # Explicitly define the target disease columns from dataset
            target_cols = [
                "HeartDisease", "Diabetes", "Hypertension", "Asthma", "KidneyDisease",
                "LiverDisease", "Cancer", "Obesity", "Arthritis", "COPD", "MentalHealthIssue"
            ]
            feature_cols = [col for col in train_df.columns if col not in target_cols]

            X_train = train_df[feature_cols]
            y_train = train_df[target_cols]
            X_test = test_df[feature_cols]
            y_test = test_df[target_cols]

            # Drop rows with NaN in targets
            logging.info("Dropping rows with NaN in target labels")
            train_mask = ~y_train.isna().any(axis=1)
            test_mask = ~y_test.isna().any(axis=1)

            X_train, y_train = X_train[train_mask], y_train[train_mask]
            X_test, y_test = X_test[test_mask], y_test[test_mask]

            # Get preprocessor
            preprocessor = self.get_data_transformer_object(X_train)

            logging.info("Applying preprocessing transformations")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Save preprocessor
            with open(self.preprocessor_path, "wb") as f:
                pickle.dump(preprocessor, f)

            logging.info(f"Preprocessor saved at: {self.preprocessor_path}")
            logging.info(f"Train Shape: {X_train_transformed.shape}, Test Shape: {X_test_transformed.shape}")

            return (
                X_train_transformed,
                X_test_transformed,
                y_train.values,
                y_test.values,
                self.preprocessor_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        obj = DataTransformation()
        X_train, X_test, y_train, y_test, preprocessor_path = obj.initiate_data_transformation(
            train_path="artifacts/train.csv",
            test_path="artifacts/test.csv"
        )

        print("✅ Data Transformation Completed")
        print(f"Preprocessor saved at: {preprocessor_path}")
        print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

    except Exception as e:
        raise CustomException(e, sys)
