from agents.watsonx_explainer import explain_result

patient = {
    "age": 63,
    "cholesterol": 233,
    "chest_pain": 3
}

result = explain_result(
    patient,
    prediction=1,
    probability=0.98
)

print(result)