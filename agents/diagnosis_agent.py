import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model_path = BASE_DIR / "models" / "heart_model.pkl"

model = joblib.load(model_path)

def predict_disease(features):
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }