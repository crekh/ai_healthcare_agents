from agents.watsonx_client import get_model

def explain_result(input_data, prediction, probability):

    model = get_model()

    prompt = f"""
    You are a healthcare AI assistant.

    Patient data:
    {input_data}

    Model prediction:
    - Heart disease: {prediction}
    - Risk probability: {probability}

    Explain this in simple, clinical but safe language.
    Do not provide diagnosis. Only interpretation.
    """

    response = model.generate(prompt)

    return response["results"][0]["generated_text"]