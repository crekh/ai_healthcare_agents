import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from agents.healthcare_orchestrator import run_healthcare_agent

st.set_page_config(
    page_title="AI Healthcare Agent",
    page_icon="🫀",
    layout="centered"
)

# ---------------------------------
# PAGE HEADER
# ---------------------------------

st.title("🫀 AI Healthcare Agent")

st.markdown(
    """
    ### IBM Watsonx + Multi-Agent Healthcare AI System

    This system combines:
    - Machine Learning prediction
    - Agentic AI orchestration
    - IBM Watsonx LLM explanations
    - Clinical recommendation generation
    """
)

st.markdown("---")

# ---------------------------------
# FEATURE LABELS / ENCODINGS
# ---------------------------------

cp_options = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

sex_options = {
    "Female": 0,
    "Male": 1
}

fbs_options = {
    "No (≤ 120 mg/dl)": 0,
    "Yes (> 120 mg/dl)": 1
}

restecg_options = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

exang_options = {
    "No": 0,
    "Yes": 1
}

slope_options = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

ca_options = {
    "0 Major Vessels": 0,
    "1 Major Vessel": 1,
    "2 Major Vessels": 2,
    "3 Major Vessels": 3
}

# ✅ FIXED THAL ENCODING
# Matches common UCI/Kaggle heart dataset

thal_options = {
    "Normal": 3,
    "Fixed Defect": 6,
    "Reversible Defect": 7
}

# ---------------------------------
# INPUT FORM
# ---------------------------------

with st.form("patient_form"):

    st.subheader("📋 Patient Information")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50
    )

    sex_label = st.selectbox(
        "Sex",
        list(sex_options.keys())
    )

    cp_label = st.selectbox(
        "Chest Pain Type",
        list(cp_options.keys())
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=80,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=100,
        max_value=600,
        value=200
    )

    fbs_label = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        list(fbs_options.keys())
    )

    restecg_label = st.selectbox(
        "Resting ECG",
        list(restecg_options.keys())
    )

    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=60,
        max_value=250,
        value=150
    )

    exang_label = st.selectbox(
        "Exercise Induced Angina",
        list(exang_options.keys())
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope_label = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        list(slope_options.keys())
    )

    ca_label = st.selectbox(
        "Number of Major Vessels",
        list(ca_options.keys())
    )

    thal_label = st.selectbox(
        "Thalassemia Test Result",
        list(thal_options.keys())
    )

    submitted = st.form_submit_button("🔍 Analyze Patient")

# ---------------------------------
# PROCESS INPUT
# ---------------------------------

if submitted:

    patient_data = {
        "age": age,
        "sex": sex_options[sex_label],
        "cp": cp_options[cp_label],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs_options[fbs_label],
        "restecg": restecg_options[restecg_label],
        "thalach": thalach,
        "exang": exang_options[exang_label],
        "oldpeak": oldpeak,
        "slope": slope_options[slope_label],
        "ca": ca_options[ca_label],
        "thal": thal_options[thal_label]
    }

    # ✅ EXPLICIT FEATURE ORDER
    # MUST match training dataset order

    features = [
        patient_data["age"],
        patient_data["sex"],
        patient_data["cp"],
        patient_data["trestbps"],
        patient_data["chol"],
        patient_data["fbs"],
        patient_data["restecg"],
        patient_data["thalach"],
        patient_data["exang"],
        patient_data["oldpeak"],
        patient_data["slope"],
        patient_data["ca"],
        patient_data["thal"]
    ]

    try:

        with st.spinner("🧠 AI agents are analyzing patient data..."):

            result = run_healthcare_agent(
                patient_data,
                features
            )

        st.success("✅ Analysis Complete")

    except Exception as e:

        st.error(f"❌ Error during analysis: {str(e)}")
        st.stop()

    st.markdown("---")

    # ---------------------------------
    # TRIAGE DISPLAY
    # ---------------------------------

    st.subheader("🟡 Triage Assessment")

    triage_level = result["triage_level"]

    if "high" in triage_level.lower():
        st.error(triage_level)

    elif "medium" in triage_level.lower():
        st.warning(triage_level)

    else:
        st.success(triage_level)

    # ---------------------------------
    # ML DIAGNOSIS
    # ---------------------------------

    st.subheader("🟠 ML Diagnosis")

    prediction = result["diagnosis"]["prediction"]
    probability = float(result["diagnosis"]["probability"])

    col1, col2 = st.columns(2)

    with col1:

        # ✅ Works with string labels
        st.metric(
            "Heart Disease Risk",
            prediction
        )

    with col2:

        st.metric(
            "Risk Probability (%)",
            f"{probability * 100:.1f}%"
        )

    st.progress(
        min(probability, 1.0),
        text=f"Risk Score: {probability:.2f}"
    )

    # ---------------------------------
    # RISK INTERPRETATION
    # ---------------------------------

    st.subheader("📊 Risk Interpretation")

    if probability >= 0.80:

        st.error(
            "High predicted cardiovascular risk"
        )

    elif probability >= 0.50:

        st.warning(
            "Moderate predicted cardiovascular risk"
        )

    else:

        st.success(
            "Lower predicted cardiovascular risk"
        )

    # ---------------------------------
    # WATSONX EXPLANATION
    # ---------------------------------

    st.subheader("🔵 AI Clinical Explanation")

    st.write(result["explanation"])

    # ---------------------------------
    # RECOMMENDATIONS
    # ---------------------------------

    st.subheader("🟢 Recommendations")

    for recommendation in result["recommendations"]:

        st.write(f"• {recommendation}")

    # ---------------------------------
    # RAW DATA
    # ---------------------------------

    with st.expander("📁 View Encoded Patient Data"):

        st.json(patient_data)

    st.markdown("---")

    st.caption(
        "Powered by IBM Watsonx + Streamlit + Multi-Agent AI Architecture"
    )