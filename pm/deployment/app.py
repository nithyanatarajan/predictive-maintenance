import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# ---------- Page Config ----------
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Custom CSS ----------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
    }
    div.stButton > button {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 0.75rem 3rem;
        border-radius: 0.5rem;
        cursor: pointer;
    }
    div.stButton > button:hover {
        background-color: #0069d9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Helper Functions ----------
def slugify(name: str) -> str:
    """Convert name to HF-compatible slug (underscores to hyphens)."""
    return name.replace("_", "-")

# ---------- Load Model (full pipeline with preprocessor) ----------
hf_username = os.getenv("HF_USERNAME")
hf_model_name = slugify(os.getenv("HF_MODEL_NAME", "predictive-maintenance-model"))

model_path = hf_hub_download(
    repo_id=f"{hf_username}/{hf_model_name}",
    filename="best_engine_maintenance_model.joblib"
)
model = joblib.load(model_path)

# Production threshold for maintenance classification
THRESHOLD = 0.4

# ---------- App Header ----------
st.title("🔧 Engine Predictive Maintenance")
st.write("Predict whether an engine requires maintenance based on sensor readings.")

st.markdown("---")

# ---------- Sensor Readings ----------
st.subheader("📊 Sensor Readings")

col1, col2, col3 = st.columns(3, gap="medium")

# Column 1 - Engine Performance
with col1:
    st.markdown("**Engine Performance**")
    engine_rpm = st.number_input("Engine RPM", min_value=0, max_value=3000, value=1500)
    fuel_pressure = st.number_input("Fuel Pressure (bar)", min_value=0.0, max_value=25.0, value=6.5, step=0.1)

# Column 2 - Lubrication System
with col2:
    st.markdown("**Lubrication System**")
    lub_oil_pressure = st.number_input("Lub Oil Pressure (bar)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
    lub_oil_temp = st.number_input("Lub Oil Temperature (°C)", min_value=0.0, max_value=150.0, value=85.0, step=1.0)

# Column 3 - Cooling System
with col3:
    st.markdown("**Cooling System**")
    coolant_pressure = st.number_input("Coolant Pressure (bar)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    coolant_temp = st.number_input("Coolant Temperature (°C)", min_value=0.0, max_value=200.0, value=90.0, step=1.0)

st.markdown("---")

# ---------- Derived Features (calculated automatically) ----------
rpm_x_fuel_pressure = engine_rpm * fuel_pressure
rpm_bins = 0 if engine_rpm < 300 else (1 if engine_rpm <= 1500 else 2)
oil_health_index = lub_oil_pressure / lub_oil_temp if lub_oil_temp > 0 else 0

# ---------- Create Input DataFrame ----------
input_data = pd.DataFrame([{
    'engine_rpm': engine_rpm,
    'lub_oil_pressure': lub_oil_pressure,
    'fuel_pressure': fuel_pressure,
    'coolant_pressure': coolant_pressure,
    'lub_oil_temp': lub_oil_temp,
    'coolant_temp': coolant_temp,
    'rpm_x_fuel_pressure': rpm_x_fuel_pressure,
    'rpm_bins': rpm_bins,
    'oil_health_index': oil_health_index
}])

# ---------- Preview & Predict ----------
st.subheader("📦 Feature Preview")
with st.expander("Click to expand (includes derived features)", expanded=False):
    cols = st.columns(3)
    for i, (field, value) in enumerate(input_data.iloc[0].items()):
        with cols[i % 3]:
            display_value = f"{value:.4f}" if isinstance(value, float) else value
            st.metric(label=field, value=display_value)

if st.button("Predict Maintenance Need"):
    # Pipeline handles preprocessing (StandardScaler)
    probability = model.predict_proba(input_data)[0, 1]
    prediction = 1 if probability >= THRESHOLD else 0

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **Maintenance Required** (Probability: {probability:.2%})")
        st.write("The engine shows signs of degradation. Schedule maintenance soon.")
    else:
        st.success(f"✅ **Normal Operation** (Probability of failure: {probability:.2%})")
        st.write("The engine is operating within normal parameters.")
        st.balloons()
