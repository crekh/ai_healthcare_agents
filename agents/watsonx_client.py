import streamlit as st
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference


def get_credentials():
    return Credentials(
        url=st.secrets["WATSONX_URL"],
        api_key=st.secrets["WATSONX_API_KEY"]
    )


def get_client():
    return APIClient(get_credentials())


def get_model():
    client = get_client()

    return ModelInference(
        model_id = "ibm/granite-4-h-small",
        api_client=client,
        project_id=st.secrets["WATSONX_PROJECT_ID"],
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 300,
            "temperature": 0.2
        }
    )


def generate_response(prompt: str):
    model = get_model()
    return model.generate_text(prompt)