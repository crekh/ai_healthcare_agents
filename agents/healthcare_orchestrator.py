from agents.triage_agent import triage_patient
from agents.diagnosis_agent import predict_disease
from agents.watsonx_explainer import explain_result
from agents.recommendation_agent import recommend

def run_healthcare_agent(patient_data, features):

    if not features:
        raise ValueError("Invalid feature input")

    # 🟡 Step 1: Triage
    triage = triage_patient(patient_data)

    # 🟠 Step 2: ML Diagnosis
    diagnosis = predict_disease(features)

    # 🔵 Step 3: LLM Explanation (Watsonx)
    explanation = explain_result(
        patient_data,
        diagnosis["prediction"],
        diagnosis["probability"]
    )

    # 🟢 Step 4: Recommendations
    recommendations = recommend(triage)

    return {
        "triage_level": triage,
        "diagnosis": {
            "prediction": diagnosis["prediction"],
            "probability": round(diagnosis["probability"], 2)
        },
        "explanation": explanation,
        "recommendations": recommendations
    }