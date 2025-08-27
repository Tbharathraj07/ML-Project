import os
import sys
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.exception import CustomException


def save_object(file_path: str, obj):
    """
    Save any Python object to disk using pickle.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    Load a Python object from disk using pickle.
    """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict):
    """
    Train and evaluate multiple models for multi-output classification.
    Returns:
        - model_report: {model_name: {"accuracy": float, "f1_score": float}}
        - trained_models: {model_name: fitted_model}
    """
    try:
        model_report = {}
        trained_models = {}

        for name, model in models.items():
            # Fit the model
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Accuracy & F1 (micro-average for multi-label)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="micro")

            model_report[name] = {
                "accuracy": acc,
                "f1_score": f1
            }
            trained_models[name] = model

        return model_report, trained_models

    except Exception as e:
        raise CustomException(e, sys)
