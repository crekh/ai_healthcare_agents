import joblib
import numpy as np

model = joblib.load("models/heart_model.pkl")

def predict_disease(features):
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }