#
# A Streamlit application for a disease risk prediction dashboard with
# a classic and elegant frontend.
#
# This version uses custom CSS and HTML embedded via st.markdown to
# create a more polished user interface, including:
# - A classic color palette with a light gray background and dark text.
# - Card-like containers for the input form and results.
# - Custom styling for buttons and input fields.
# - A visually enhanced display for the prediction results.
#
# The core machine learning logic for loading the model and making predictions
# remains the same, but the presentation layer is significantly improved.
#

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.multioutput import MultiOutputClassifier

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Disease Risk Predictor",
    page_icon="🧬",
    layout="wide"
)

# -------------------------------
# Custom CSS for classic look
# -------------------------------
# We're using st.markdown with unsafe_allow_html=True to inject
# our custom CSS and HTML. This gives us full control over the styling.
custom_css = """
<style>
    /* Main body and container styling */
    body {
        font-family: 'Inter', sans-serif;
        background-color: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .classic-container {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-top: 1rem;
    }
    h1, h2, h3 {
        color: #1e1e1e;
    }
    .stSelectbox, .stNumberInput, .stTextarea, .stMarkdown, .stButton > button {
        border-radius: 8px;
    }
    
    /* Custom button styling */
    .stButton > button {
        background-color: #2c3e50; /* A classic, dark blue-gray */
        color: white;
        padding: 12px 24px;
        font-weight: bold;
        transition: background-color 0.3s ease, transform 0.2s ease;
        border: none;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #34495e;
        transform: translateY(-2px);
    }
    
    /* Result card styling */
    .result-card {
        background-color: #f9f9f9;
        border-left: 5px solid; /* Will be colored with JS */
        border-radius: 8px;
        padding: 15px 20px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: transform 0.2s ease;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    }
    .result-card:hover {
        transform: translateX(5px);
    }
    .risk-high {
        border-color: #e74c3c; /* Red for high risk */
    }
    .risk-low {
        border-color: #2ecc71; /* Green for low risk */
    }
    .risk-label {
        font-weight: 600;
        font-size: 1.1rem;
        color: #333;
    }
    .risk-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #555;
    }
    .risk-status {
        font-weight: bold;
        text-transform: uppercase;
        font-size: 0.9rem;
    }
    .status-high { color: #e74c3c; }
    .status-low { color: #2ecc71; }

    /* Targeting Streamlit-specific elements to remove padding */
    .stForm, .stEmpty {
        padding: 0 !important;
        margin: 0 !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# -------------------------------
# Load Model & Preprocessor
# -------------------------------
@st.cache_resource
def load_artifacts():
    """
    Loads the pre-trained model and preprocessor artifacts.
    Uses st.cache_resource to cache the artifacts for efficiency.
    """
    try:
        if not os.path.exists("artifacts/model.pkl") or not os.path.exists("artifacts/preprocessor.pkl"):
            st.error("Model or preprocessor files not found. Please ensure 'artifacts/model.pkl' and 'artifacts/preprocessor.pkl' exist.")
            st.stop()
            
        artifacts = joblib.load("artifacts/model.pkl")   # {"model": ..., "thresholds": [...]}
        model = artifacts["model"]
        thresholds = artifacts["thresholds"]
        preprocessor = joblib.load("artifacts/preprocessor.pkl")
        return model, thresholds, preprocessor
    except Exception as e:
        st.error(f"Failed to load model artifacts. Error: {e}")
        st.stop()


model, thresholds, preprocessor = load_artifacts()

# -------------------------------
# Disease Names
# -------------------------------
disease_names = [
    "Diabetes", "Hypertension", "Heart Disease", "Stroke", "Obesity",
    "Kidney Disease", "Liver Disease", "Asthma", "Arthritis",
    "COPD", "Cancer"
]

# -------------------------------
# Prediction Function
# -------------------------------
def predict_risk(input_df):
    """
    Performs the prediction and calculates risk probabilities.
    
    Args:
        input_df (pd.DataFrame): The user's input data as a DataFrame.
        
    Returns:
        tuple: A tuple containing the risk percentages and binary predictions.
    """
    try:
        # Preprocess input
        X = preprocessor.transform(input_df)
    
        # Predict probabilities
        if isinstance(model, MultiOutputClassifier):
            proba = np.hstack([
                est.predict_proba(X)[:, 1].reshape(-1, 1) for est in model.estimators_
            ])
        elif hasattr(model, "predict_proba"):
            raw_proba = model.predict_proba(X)
            if isinstance(raw_proba, list):
                proba = np.hstack([p[:, 1].reshape(-1, 1) for p in raw_proba])
            else:
                proba = raw_proba
        else:
            st.error("⚠️ Model does not support probability prediction.")
            return None, None
            
        # Convert to percentages
        risk_percentages = (proba[0] * 100).round(2)
        
        # Apply thresholds to get binary predictions
        binary_preds = (proba[0] >= np.array(thresholds)).astype(int)
        
        return risk_percentages, binary_preds
        
    except Exception as e:
        st.error(f"⚠️ Prediction failed. Please check your inputs. Error: {e}")
        return None, None

# -------------------------------
# App Layout
# -------------------------------
st.title("🩺 Disease Risk Prediction Dashboard")
st.markdown("Enter your health details below to estimate your **disease risk probabilities (%)**.")

st.markdown('<div class="classic-container">', unsafe_allow_html=True)
st.subheader("👤 Enter Your Health Information")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30, help="Your age in years.")
        sex = st.selectbox("Sex", ["Male", "Female"], help="Your biological sex.")
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0, help="Body Mass Index (kg/m^2).")
        blood_pressure = st.number_input("Blood Pressure (Systolic)", min_value=80, max_value=200, value=115, help="Systolic blood pressure (mmHg).")
        cholesterol_level = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=160, help="Total cholesterol level (mg/dL).")

    with col2:
        glucose_level = st.number_input("Glucose Level", min_value=50, max_value=300, value=90, help="Fasting glucose level (mg/dL).")
        physical_activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"], index=2, help="Level of physical activity.")
        diet_quality = st.selectbox("Diet Quality", ["Poor", "Average", "Good"], index=2, help="Quality of your diet.")
        alcohol_consumption = st.selectbox("Alcohol Consumption", ["Yes", "No"], index=1, help="Do you consume alcohol?")
        smoker = st.selectbox("Smoker", ["Yes", "No"], index=1, help="Are you a smoker?")
        family_history = st.selectbox("Family History of Disease", ["Yes", "No"], index=1, help="Any family history of chronic diseases?")
        mental_health_issue = st.selectbox("Mental Health Issue", ["Yes", "No"], index=1, help="Any diagnosed mental health issues?")

    submit_button = st.form_submit_button("🔍 Predict Risk")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Display Prediction Results
# -------------------------------
if submit_button:
    input_df = pd.DataFrame([{
        "Age": age, "Sex": sex, "BMI": bmi, "BloodPressure": blood_pressure,
        "CholesterolLevel": cholesterol_level, "GlucoseLevel": glucose_level,
        "PhysicalActivity": physical_activity, "DietQuality": diet_quality,
        "AlcoholConsumption": alcohol_consumption, "Smoker": smoker,
        "FamilyHistory": family_history, "MentalHealthIssue": mental_health_issue,
    }])
    
    risk_percentages, binary_preds = predict_risk(input_df)

    if risk_percentages is not None:
        st.markdown('<div class="classic-container">', unsafe_allow_html=True)
        st.subheader("📊 Prediction Results")
        
        st.markdown(
            "The dashboard has analyzed your health information to provide a risk assessment. "
            "Here are the estimated probabilities for various diseases:"
        )
        
        # Display results using custom HTML and CSS
        num_cols = 3
        cols = st.columns(num_cols)
        
        for idx, (disease, prob) in enumerate(zip(disease_names, risk_percentages)):
            col_idx = idx % num_cols
            with cols[col_idx]:
                status_text = "Low Risk" if binary_preds[idx] == 0 else "High Risk"
                status_class = "status-low" if binary_preds[idx] == 0 else "status-high"
                card_class = "risk-low" if binary_preds[idx] == 0 else "risk-high"
                
                # Use custom HTML for the result card
                st.markdown(f"""
                <div class="result-card {card_class}">
                    <div>
                        <div class="risk-label">{disease}</div>
                        <div class="risk-status {status_class}">{status_text}</div>
                    </div>
                    <div class="risk-value">{prob}%</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
