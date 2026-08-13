# ------------------------------------------------------------------
# Streamlit dashboard for the ClinTox toxicity / FDA approval
# predictor. Sends the entered SMILES string to the FastAPI service
# and displays the prediction results in a readable format.
# ------------------------------------------------------------------

import streamlit as st
import requests
import os

# Inside Docker, the dashboard talks to the API via the service name
# "api" on the internal Docker network. When running locally without
# Docker, this should point to localhost instead.
API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(page_title="ClinTox Toxicity Predictor", page_icon="🧪")

st.title("🧪 Molecular Toxicity & FDA Approval Predictor")
st.markdown(
    "Enter a molecule's SMILES string to predict its clinical toxicity risk "
    "and likelihood of FDA approval, based on a Random Forest model trained "
    "on the ClinTox dataset."
)

smiles_input = st.text_input(
    "SMILES string",
    value="CC(=O)OC1=CC=CC=C1C(=O)O",
    help="Example: CC(=O)OC1=CC=CC=C1C(=O)O (aspirin)"
)

if st.button("Predict"):
    if not smiles_input.strip():
        st.warning("Please enter a SMILES string.")
    else:
        with st.spinner("Running prediction..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"smiles": smiles_input},
                    timeout=10
                )
            except requests.exceptions.ConnectionError:
                st.error("Could not reach the prediction API. Is it running?")
                st.stop()

        if response.status_code == 200:
            result = response.json()

            col1, col2 = st.columns(2)
            col1.metric("FDA Approval Probability", f"{result['fda_approved_prob']:.1%}")
            col2.metric("Clinical Toxicity Probability", f"{result['ct_tox_prob']:.1%}")

            if result["aromatic_amine_alert"]:
                st.warning(
                    "⚠️ Structural alert: this molecule contains an aromatic amine "
                    "substructure, associated with elevated toxicity risk in "
                    "medicinal chemistry literature."
                )
            else:
                st.success("No aromatic amine structural alert detected.")

            st.caption(
                "Note: predictions reflect patterns in the ClinTox training "
                "dataset, not verified clinical ground truth. Not medical advice."
            )
        else:
            st.error(f"Invalid SMILES string — could not parse molecule.")

st.divider()
st.caption("Recent predictions are logged to the project's PostgreSQL database.")

# ------------------------------------------------------------------
# Ask the agent (Ollama + tool-calling) a natural-language question.
# ------------------------------------------------------------------

st.divider()
st.header("💬 Ask the Assistant")
st.markdown(
    "Ask a question in plain language — e.g. *\"Is aspirin likely to be "
    "toxic?\"* or *\"Is caffeine FDA approved?\"*. The assistant can look "
    "up compounds by name and run predictions automatically."
)

question_input = st.text_input(
    "Your question",
    value="Is aspirin likely to be toxic?",
    key="agent_question"
)

if st.button("Ask", key="ask_button"):
    if not question_input.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                ask_response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question_input},
                    timeout=60
                )
            except requests.exceptions.ConnectionError:
                st.error("Could not reach the assistant API. Is it running?")
                st.stop()

        if ask_response.status_code == 200:
            answer = ask_response.json().get("answer", "")
            st.markdown(answer)
        else:
            st.error("The assistant could not process this question.")