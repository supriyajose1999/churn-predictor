"""
Streamlit front-end for the Customer Churn Prediction API.

This is a thin UI layer -- it does NOT run the model itself. It sends the
customer details you enter to the FastAPI backend (running locally or
deployed on Render/Railway) and displays the response.

Run locally:
    streamlit run streamlit_app/app.py

Requires the FastAPI backend to be running (see main project README).
Set the backend URL via the API_URL environment variable, or edit the
default below, or (once deployed) set it as a Streamlit Cloud secret.
"""

import os
import requests
import streamlit as st

# --- Config ---
DEFAULT_API_URL = "http://localhost:8000"

def _get_api_url():
    # st.secrets throws an error (rather than returning None) if no
    # secrets.toml file exists at all, which is the normal case for local
    # development -- so we guard against that instead of using .get() directly.
    try:
        if "API_URL" in st.secrets:
            return st.secrets["API_URL"]
    except Exception:
        pass
    return os.environ.get("API_URL", DEFAULT_API_URL)

API_URL = _get_api_url()

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="centered",
)

st.title("📉 Customer Churn Predictor")
st.caption(
    "Enter a customer's account details to estimate their churn risk. "
    "This UI calls a live FastAPI model-serving backend."
)

with st.expander("ℹ️ About this tool", expanded=False):
    st.write(
        "This predicts the probability that a telecom customer will cancel their "
        "subscription, based on a RandomForest model trained on customer account data. "
        "It's a demo built on synthetic data -- see the full project README for details "
        "on validation, limitations, and the tech stack."
    )
    st.code(f"Backend API: {API_URL}", language=None)

st.divider()

# --- Input form ---
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure_months = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

with col2:
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )

monthly_charges = st.slider("Monthly Charges ($)", 18.0, 130.0, 70.0, step=0.5)
total_charges = st.slider("Total Charges ($)", 0.0, 9000.0, float(monthly_charges * tenure_months), step=10.0)

st.divider()

# --- Quick-fill presets, so a recruiter can test in one click without typing ---
preset_col1, preset_col2, _ = st.columns([1, 1, 2])
with preset_col1:
    st.caption("Try a preset:")
with preset_col2:
    pass

payload = {
    "gender": gender,
    "senior_citizen": senior_citizen,
    "partner": partner,
    "dependents": dependents,
    "tenure_months": tenure_months,
    "phone_service": phone_service,
    "multiple_lines": multiple_lines,
    "internet_service": internet_service,
    "online_security": online_security,
    "online_backup": online_backup,
    "device_protection": device_protection,
    "tech_support": tech_support,
    "streaming_tv": streaming_tv,
    "streaming_movies": streaming_movies,
    "contract": contract,
    "paperless_billing": paperless_billing,
    "payment_method": payment_method,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
}

if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
    try:
        with st.spinner("Contacting the model API..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            prob = result["churn_probability"]
            tier = result["risk_tier"]
            prediction = result["churn_prediction"]

            st.subheader("Result")
            tier_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            st.metric("Churn Probability", f"{prob * 100:.1f}%")
            st.markdown(f"### {tier_colors.get(tier, '⚪')} Risk Tier: **{tier}**")
            st.progress(min(prob, 1.0))

            if prediction == "Yes":
                st.warning(
                    "This customer is predicted to **churn**. Consider a retention offer "
                    "or proactive outreach."
                )
            else:
                st.success("This customer is predicted to **stay**.")

            with st.expander("Raw API response"):
                st.json(result)
        else:
            st.error(f"API returned an error (status {response.status_code}): {response.text}")
    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not reach the API at `{API_URL}`. Make sure the FastAPI backend is running "
            f"(`uvicorn app.main:app --port 8000`) or that API_URL is set correctly if deployed."
        )
    except requests.exceptions.Timeout:
        st.error("The API request timed out. If using a free-tier deployment, it may be waking up from idle -- try again in a few seconds.")

st.divider()
st.caption("Built with FastAPI (model serving) + Streamlit (this UI) + scikit-learn (model).")
