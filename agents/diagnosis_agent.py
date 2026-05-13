import joblib
import pandas as pd
from pathlib import Path

# ---------------------------------
# BASE DIRECTORY
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------
# MODEL PATHS
# ---------------------------------

model_path = BASE_DIR / "models" / "heart_model.pkl"
scaler_path = BASE_DIR / "models" / "scaler.pkl"

# ---------------------------------
# LOAD ARTIFACTS
# ---------------------------------

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ---------------------------------
# FEATURE ORDER
# MUST MATCH TRAINING DATASET
# ---------------------------------

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

# ---------------------------------
# PREDICTION FUNCTION
# ---------------------------------

def predict_disease(features):

    # -----------------------------
    # VALIDATION
    # -----------------------------

    if len(features) != len(FEATURE_COLUMNS):

        raise ValueError(
            f"Expected {len(FEATURE_COLUMNS)} features, got {len(features)}"
        )

    # -----------------------------
    # CREATE DATAFRAME
    # -----------------------------

    features_df = pd.DataFrame(
        [features],
        columns=FEATURE_COLUMNS
    )

    # -----------------------------
    # SCALE FEATURES
    # -----------------------------

    features_scaled = scaler.transform(features_df)

    # -----------------------------
    # PREDICT PROBABILITY
    # -----------------------------

    probability = model.predict_proba(features_scaled)[0][1]

    # -----------------------------
    # CUSTOM THRESHOLD
    # More sensitive for healthcare
    # -----------------------------

    prediction = 1 if probability >= 0.40 else 0

    # -----------------------------
    # HUMAN READABLE LABEL
    # -----------------------------

    risk_label = (
        "High Risk"
        if prediction == 1
        else "Low Risk"
    )

    # -----------------------------
    # DEBUG LOGGING
    # Optional for testing
    # -----------------------------

    print("\n======================")
    print("INPUT FEATURES:")
    print(features_df)

    print("\nRISK PROBABILITY:")
    print(probability)

    print("\nPREDICTION:")
    print(risk_label)
    print("======================\n")

    # -----------------------------
    # RETURN RESULTS
    # -----------------------------

    return {
        "prediction": risk_label,
        "probability": round(float(probability), 2)
    }