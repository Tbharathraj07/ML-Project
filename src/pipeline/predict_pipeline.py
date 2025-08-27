import numpy as np
import joblib
import sys
from src.exception import CustomException

def predict_patient(new_data):
    try:
        # 1️⃣ Load saved object
        saved_obj = joblib.load("artifacts/model.pkl")

        model = saved_obj["model"]
        thresholds = saved_obj.get("thresholds", None)

        # 2️⃣ Define your disease target columns (must match training!)
        target_columns = [
            "HeartDisease", "Diabetes", "Hypertension", "Asthma", "KidneyDisease",
            "LiverDisease", "Cancer", "Obesity", "Arthritis", "COPD", "MentalHealthIssue"
        ]

        # 3️⃣ Predict probabilities
        probabilities = np.array([
            clf.predict_proba(new_data)[:, 1]  # prob of class 1
            for clf in model.estimators_
        ]).T  # shape = (n_samples, n_targets)

        # 4️⃣ Apply thresholds (if available, else 0.5 default)
        if thresholds is not None:
            predictions = (probabilities >= thresholds).astype(int)
        else:
            predictions = (probabilities >= 0.5).astype(int)

        # 5️⃣ Format results
        risk_report = {
            disease: {
                "probability": f"{prob*100:.2f}%",
                "prediction": "Yes" if pred == 1 else "No"
            }
            for disease, prob, pred in zip(target_columns, probabilities[0], predictions[0])
        }

        return risk_report

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    # Example random patient with 22 features
    new_patient = np.random.rand(1, 22)

    print("🔮 Predicting Disease Risks for New Patient...\n")
    report = predict_patient(new_patient)

    for disease, result in report.items():
        print(f"{disease}: {result['probability']} → {result['prediction']}")
