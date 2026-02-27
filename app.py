import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Vehicle Maintenance Predictor", layout="centered")

# ----------------------------
# Constants & Typical Values (from training data)
# ----------------------------
# Derived from our audit analysis of the 50,000 row training set
TYPICAL_VALUES = {
    "Mileage": 54931.2,
    "Reported_Issues": 2.5,
    "Vehicle_Age": 5.5,
    "Service_History": 5.5,
    "Fuel_Efficiency": 15.0,
    "Insurance_Premium": 17465.3,
    "Last_Service_Days_Ago": 896.6,
    "Warranty_Days_Remaining": -320.3
}

# ----------------------------
# Load artifacts
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "maintenance_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")

@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    except Exception as e:
        return None, None

model, preprocessor = load_model_artifacts()

if model is None:
    st.error("⚠️ Model artifacts missing. Please run `python train.py` first.")
    st.stop()

# ----------------------------
# Header
# ----------------------------
st.title("🚗 Vehicle Maintenance Prediction")
st.write("Enter vehicle details to predict whether maintenance is needed.")
st.info(
    "ℹ️ This model is trained on **synthetic data**. All inputs are constrained "
    "to in-distribution ranges observed during training for prediction stability.",
    icon="🔬",
)

st.divider()

# ============================================================
# INPUTS
# ============================================================

# ---------- Vehicle Specs ----------
st.subheader("🔧 Vehicle Specifications")
col1, col2 = st.columns(2)

with col1:
    Vehicle_Model = st.selectbox(
        "Vehicle Model", ["Car", "SUV", "Van", "Truck", "Bus", "Motorcycle"]
    )
    Engine_Size = st.selectbox("Engine Size (cc)", [800, 1000, 1500, 2000, 2500], index=2)
    Transmission_Type = st.selectbox("Transmission Type", ["Automatic", "Manual"])

with col2:
    Vehicle_Age = st.number_input("Vehicle Age (years)", 1, 10, 5, 1)
    Fuel_Type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric"])
    Owner_Type = st.selectbox("Owner Type", ["First", "Second", "Third"])

# ---------- Usage & History ----------
st.subheader("📏 Usage & Service History")
col3, col4 = st.columns(2)

with col3:
    Mileage = st.number_input("Total Mileage (km)", 30000, 80000, 55000, 5000)
    Service_History = st.number_input("Number of Past Services", 1, 10, 5, 1)
    Last_Service_Days_Ago = st.number_input("Days Since Last Service", 700, 1100, 896, 10)

with col4:
    Odometer_Reading = st.number_input("Odometer Reading (km)", 1000, 150000, 75000, 5000)
    Reported_Issues = st.selectbox("Reported Issues", [0, 1, 2, 3, 4, 5], index=2)
    Warranty_Days_Remaining = st.number_input("Warranty Days Remaining", -700, 50, -320, 10)

col5, col6 = st.columns(2)
with col5:
    Fuel_Efficiency = st.number_input("Fuel Efficiency (km/l)", 10.0, 20.0, 15.0, 0.5)
with col6:
    Insurance_Premium = st.number_input("Annual Insurance Premium (₹)", 5000, 30000, 17500, 1000)

# ---------- Component Condition ----------
st.subheader("🔋 Component Condition")
col7, col8, col9 = st.columns(3)

with col7:
    Tire_Condition = st.selectbox("Tire Condition", ["New", "Good", "Worn Out"])
with col8:
    Brake_Condition = st.selectbox("Brake Condition", ["New", "Good", "Worn Out"])
with col9:
    Battery_Status = st.selectbox("Battery Status", ["New", "Good", "Weak"])

Maintenance_History = st.selectbox("Overall Maintenance History", ["Good", "Average", "Poor"], index=1)
Accident_History = st.selectbox("Accident History (count)", [0, 1, 2, 3], index=1)

# ============================================================
# ANALYSIS WIDGETS
# ============================================================
st.divider()

# Comparison Feature Snapshot Chart
st.subheader("📊 Input Comparison")
st.write("How your inputs compare to typical values of the dataset:")

# Prepare comparison data
comparison_features = ["Vehicle_Age", "Reported_Issues", "Service_History", "Fuel_Efficiency"]
user_vals = [Vehicle_Age, Reported_Issues, Service_History, Fuel_Efficiency]
typical_vals = [TYPICAL_VALUES[f] for f in comparison_features]

comp_df = pd.DataFrame({
    "Feature": ["Age", "Issues", "Services", "Efficiency"],
    "Your Input": user_vals,
    "Dataset Typical": typical_vals
}).set_index("Feature")

st.bar_chart(comp_df)
st.caption("Lower Age/Issues/Services generally reduces risk, while higher Efficiency is better.")

# ============================================================
# PREDICT
# ============================================================
st.divider()

if st.button("🔍 Predict Maintenance Need", use_container_width=True):
    input_df = pd.DataFrame([{
        "Mileage": Mileage,
        "Reported_Issues": Reported_Issues,
        "Vehicle_Age": Vehicle_Age,
        "Engine_Size": Engine_Size,
        "Odometer_Reading": Odometer_Reading,
        "Insurance_Premium": float(Insurance_Premium),
        "Service_History": Service_History,
        "Accident_History": Accident_History,
        "Fuel_Efficiency": Fuel_Efficiency,
        "Last_Service_Days_Ago": Last_Service_Days_Ago,
        "Warranty_Days_Remaining": Warranty_Days_Remaining,
        "Vehicle_Model": Vehicle_Model,
        "Maintenance_History": Maintenance_History,
        "Fuel_Type": Fuel_Type,
        "Transmission_Type": Transmission_Type,
        "Owner_Type": Owner_Type,
        "Tire_Condition": Tire_Condition,
        "Brake_Condition": Brake_Condition,
        "Battery_Status": Battery_Status,
    }])

    try:
        X_transformed = preprocessor.transform(input_df)
        prob = model.predict_proba(X_transformed)[0][1]
        
        # Result Header
        st.divider()
        if prob >= 0.5:
            st.error(f"⚠️ **Needs Maintenance** (Risk: {prob:.2f})")
        else:
            st.success(f"✅ **No Maintenance Needed** (Risk: {prob:.2f})")

        # Probability Explanation Chart
        st.subheader("💡 Prediction Confidence")
        prob_chart_df = pd.DataFrame({
            "Classification": ["No Maintenance", "Needs Maintenance"],
            "Probability": [1 - prob, prob]
        }).set_index("Classification")
        
        st.bar_chart(prob_chart_df)
        st.caption("This chart shows the model's confidence in each classification outcome.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ============================================================
# DATASET OVERVIEW (Collapsible)
# ============================================================
st.divider()
with st.expander("📈 Training Dataset Overview"):
    st.write("Understand the data context the model was trained on.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**Maintenance Need Distribution**")
        # Class distribution (from previous audit: 66.8% maintenance needed)
        class_df = pd.DataFrame({
            "Class": ["Needs Maintenance", "No Maintenance"],
            "Percent": [66.8, 33.2]
        }).set_index("Class")
        st.bar_chart(class_df)
        st.caption("The training data has more examples of vehicles needing maintenance.")

    with col_b:
        st.write("**Age vs Issues Spread**")
        st.write("Training Range: 1-10 years age.")
        st.write("Typical Issues: 2-3 per vehicle.")
        # Just a placeholder caption for context since we aren't loading the full dataset for performance
        st.info("The model performs best within these typical operating ranges.")