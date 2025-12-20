# streamlit_app.py
"""
Streamlit Web Interface for the Multi-Task Learning Model.
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import torch
import numpy as np

# Import our custom modules
import importlib.util
from pathlib import Path

# Import numbered modules (Python can't import modules starting with numbers directly)
_package_dir = Path(__file__).parent

# Import config
_config_spec = importlib.util.spec_from_file_location("config", _package_dir / "01_config.py")
config = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(config)

# Import model
_model_spec = importlib.util.spec_from_file_location("model", _package_dir / "03_model.py")
model = importlib.util.module_from_spec(_model_spec)
_model_spec.loader.exec_module(model)

# Access class from loaded module
SharedBottomMTL = model.SharedBottomMTL


# ===== LOAD MODEL =====
@st.cache_resource
def load_model():
    """Load the trained model (cached to avoid reloading on every interaction)."""
    num_continuous = len(config.CONT_COLS)
    model = SharedBottomMTL(num_continuous=num_continuous, hidden_dim=config.HIDDEN_DIM)
    model.load_state_dict(torch.load('trained_model.pth', map_location='cpu'))
    model.eval()
    return model


# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Health Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

# ===== HEADER =====
st.title("🏥 Multi-Task Health Risk Prediction")
st.markdown("""
This app uses a Multi-Task Learning neural network to predict:
- **Cardiovascular Risk** (binary classification)
- **Metabolic Syndrome Components** (5 risk factors)
- **Kidney Function (ACR Log)** (regression)
- **Liver Function (ALT Log)** (regression)
""")

st.divider()

# ===== SIDEBAR INPUTS =====
st.sidebar.header("📋 Enter Patient Data")

# Demographics
st.sidebar.subheader("Demographics")
age = st.sidebar.slider("Age (RIDAGEYR)", 18, 100, 45)
pir = st.sidebar.slider("Poverty-Income Ratio (INDFMPIR)", 0.0, 5.0, 2.0, step=0.1)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
race = st.sidebar.selectbox("Race/Ethnicity", [
    "Mexican American (1)",
    "Other Hispanic (2)",
    "Non-Hispanic White (3)",
    "Non-Hispanic Black (4)",
    "Non-Hispanic Asian (6)",
    "Other/Multi-Racial (7)"
])

# Body Measures
st.sidebar.subheader("Body Measures")
bmi = st.sidebar.slider("BMI (BMXBMI)", 15.0, 60.0, 25.0, step=0.1)
height = st.sidebar.slider("Height cm (BMXHT)", 100.0, 220.0, 170.0, step=0.5)

# Vitals
st.sidebar.subheader("Vitals")
pulse = st.sidebar.slider("Pulse (BPM)", 40, 140, 70)

# Blood Count
st.sidebar.subheader("Blood Count")
wbc = st.sidebar.slider("White Blood Cells (LBXWBCSI)", 2.0, 20.0, 7.0, step=0.1)
platelets = st.sidebar.slider("Platelets (LBXPLTSI)", 100.0, 500.0, 250.0, step=5.0)
hemoglobin = st.sidebar.slider("Hemoglobin (LBXHGB)", 8.0, 20.0, 14.0, step=0.1)
mcv = st.sidebar.slider("Mean Corpuscular Volume (LBXMCVSI)", 70.0, 110.0, 90.0, step=0.5)

# Biochemistry
st.sidebar.subheader("Biochemistry")
creatinine = st.sidebar.slider("Serum Creatinine (LBXSCR)", 0.3, 5.0, 1.0, step=0.05)
ast = st.sidebar.slider("AST (LBXSASSI)", 5.0, 150.0, 25.0, step=1.0)
bilirubin = st.sidebar.slider("Total Bilirubin (LBXSTB)", 0.1, 5.0, 0.8, step=0.1)
ggt = st.sidebar.slider("GGT (LBXSGTSI)", 5.0, 200.0, 25.0, step=1.0)
uric_acid = st.sidebar.slider("Uric Acid (LBXSUA)", 2.0, 12.0, 5.5, step=0.1)
sodium = st.sidebar.slider("Sodium (LBXSNASI)", 130.0, 150.0, 140.0, step=0.5)
potassium = st.sidebar.slider("Potassium (LBXSKSI)", 3.0, 6.0, 4.0, step=0.1)
cholesterol = st.sidebar.slider("Total Cholesterol (LBXTC)", 100.0, 350.0, 200.0, step=5.0)

# Lifestyle
st.sidebar.subheader("Lifestyle")
alcohol = st.sidebar.slider("Alcohol Drinks/Week", 0.0, 50.0, 3.0, step=0.5)
smoking = st.sidebar.slider("Smoking Status (SMQ040)", 0.0, 3.0, 0.0, step=1.0)


# ===== ENCODE INPUTS =====
def encode_gender(g):
    """One-hot encode gender."""
    return [1.0, 0.0] if g == "Male" else [0.0, 1.0]


def encode_race(r):
    """One-hot encode race/ethnicity."""
    race_map = {
        "Mexican American (1)": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Other Hispanic (2)": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "Non-Hispanic White (3)": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "Non-Hispanic Black (4)": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "Non-Hispanic Asian (6)": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "Other/Multi-Racial (7)": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    }
    return race_map[r]


# ===== BUILD INPUT TENSOR =====
def build_input_tensor():
    """Construct the input tensor from user inputs."""
    # Match the order in config.CONT_COLS
    features = [
        # Demographics
        float(age), float(pir),
        # Body Measures
        float(bmi), float(height),
        # Vitals
        float(pulse),
        # Blood Count
        float(wbc), float(platelets), float(hemoglobin), float(mcv),
        # Biochemistry
        float(creatinine), float(ast), float(bilirubin), float(ggt),
        float(uric_acid), float(sodium), float(potassium), float(cholesterol),
        # Lifestyle
        float(alcohol), float(smoking),
        # One-hot encoded gender
        *encode_gender(gender),
        # One-hot encoded race
        *encode_race(race)
    ]
    return torch.tensor([features], dtype=torch.float32)


# ===== PREDICTION =====
if st.sidebar.button("🔍 Predict Risk", type="primary", use_container_width=True):
    # Load model
    model = load_model()
    
    # Build input
    x = build_input_tensor()
    
    # Make predictions
    with torch.no_grad():
        p_cardio, p_metabolic, p_kidney, p_liver = model(x)
    
    # Process outputs
    cardio_prob = torch.sigmoid(p_cardio).item()
    metabolic_probs = torch.sigmoid(p_metabolic).squeeze().numpy()
    kidney_value = p_kidney.item()
    liver_value = p_liver.item()
    
    # ===== DISPLAY RESULTS =====
    st.header("📊 Prediction Results")
    
    # Create 4 columns for the 4 tasks
    col1, col2 = st.columns(2)
    
    # Cardiovascular Risk
    with col1:
        st.subheader("❤️ Cardiovascular Risk")
        risk_level = "High" if cardio_prob > 0.5 else "Low"
        risk_color = "🔴" if cardio_prob > 0.5 else "🟢"
        st.metric(
            label="Risk Probability",
            value=f"{cardio_prob:.1%}",
            delta=f"{risk_color} {risk_level} Risk"
        )
        st.progress(cardio_prob)
    
    # Metabolic Syndrome
    with col2:
        st.subheader("⚡ Metabolic Syndrome Components")
        labels = ["Waist", "Triglycerides", "HDL", "Blood Pressure", "Glucose"]
        for label, prob in zip(labels, metabolic_probs):
            status = "⚠️" if prob > 0.5 else "✅"
            st.write(f"{status} **{label}**: {prob:.1%}")
            st.progress(float(prob))
    
    st.divider()
    
    col3, col4 = st.columns(2)
    
    # Kidney Function
    with col3:
        st.subheader("🫘 Kidney Function (ACR Log)")
        st.metric(label="Predicted ACR (Log Scale)", value=f"{kidney_value:.3f}")
        acr_actual = np.exp(kidney_value)
        st.info(f"Estimated ACR: {acr_actual:.2f} mg/g")
    
    # Liver Function
    with col4:
        st.subheader("🫀 Liver Function (ALT Log)")
        st.metric(label="Predicted ALT (Log Scale)", value=f"{liver_value:.3f}")
        alt_actual = np.exp(liver_value)
        st.info(f"Estimated ALT: {alt_actual:.2f} U/L")

else:
    # Initial state
    st.info("👈 Enter patient data in the sidebar and click **Predict Risk** to see predictions.")

# ===== FOOTER =====
st.divider()
st.caption("⚕️ This is a demonstration model. Not for clinical use.")
