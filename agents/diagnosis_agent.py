import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model_path = BASE_DIR / "models" / "heart_model.pkl"
scaler_path = BASE_DIR / "models" / "scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


def predict_disease(features):
    if len(features) != model.n_features_in_:
        raise ValueError(
            f"Expected {model.n_features_in_} features, got {len(features)}"
        )

    features_scaled = scaler.transform(pd.DataFrame([features], columns=FEATURE_COLUMNS))
    probability = model.predict_proba(features_scaled)[0][1]

    prediction = 1 if probability >= 0.40 else 0

    risk_label = "High Risk" if prediction == 1 else "Low Risk"

    return {
        "prediction": risk_label,
        "probability": round(float(probability), 2)
    }