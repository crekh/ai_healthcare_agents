from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from tools.data_loader import load_data
from tools.model_utils import save_model

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import cross_val_score

import joblib

# -----------------------------
# LOAD DATA
# -----------------------------

df = load_data("data/heart.csv")

# -----------------------------
# FEATURES / TARGET
# -----------------------------

X = df.drop("target", axis=1)

#Flip labels so:
#1 = disease
#0 = healthy

y = 1 - df["target"]

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# SCALING
# -----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# -----------------------------
# MODEL TRAINING
# -----------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

# -----------------------------
# EVALUATION
# -----------------------------

predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy:.2f}")

# -----------------------------
# CONFUSION MATRIX
# -----------------------------

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------

report = classification_report(y_test, predictions)

print("\nClassification Report:")
print(report)

# -----------------------------
# CROSS VALIDATION
# -----------------------------

cv_scores = cross_val_score(
    model,
    X_train_scaled,
    y_train,
    cv=5
)

print("\nCross Validation Scores:")
print(cv_scores)

print(f"\nAverage CV Score: {cv_scores.mean():.2f}")

# -----------------------------
# SAVE ARTIFACTS
# -----------------------------

save_model(model, "models/heart_model.pkl")

joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model and scaler saved successfully.")