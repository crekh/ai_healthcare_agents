def recommend(risk_level):

    if risk_level == "HIGH":
        return [
            "Consult a cardiologist immediately",
            "Take ECG and stress test",
            "Avoid physical exertion"
        ]

    elif risk_level == "MEDIUM":
        return [
            "Schedule doctor visit within 1 week",
            "Monitor blood pressure daily",
            "Improve diet and exercise"
        ]

    else:
        return [
            "Maintain healthy lifestyle",
            "Regular annual check-up",
            "Light exercise recommended"
        ]