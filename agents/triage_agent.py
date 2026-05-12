def triage_patient(data):
    risk_score = 0

    if data["cp"] > 2:
        risk_score += 2
    if data["age"] > 60:
        risk_score += 1
    if data["chol"] > 240:
        risk_score += 1
    if data["exang"] == 1:
        risk_score += 2

    if risk_score >= 4:
        return "HIGH"
    elif risk_score >= 2:
        return "MEDIUM"
    else:
        return "LOW"