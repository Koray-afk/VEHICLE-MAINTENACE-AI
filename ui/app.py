import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.preprocessing import get_preprocessing_pipeline, preprocess_data
from ml.train import train_models
from ml.evaluate import evaluate_model, get_feature_importance

# Set Page Config
st.set_page_config(page_title="Vehicle Maintenance AI", page_icon="🚗", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_stdio=True)

st.title("🚗 Vehicle Maintenance Prediction System")
st.markdown("---")

# Sidebar for Upload
st.sidebar.header("Data Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Vehicle Telemetry CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Feature Selection & Target Identification
    if 'maintenance_required' not in df.columns:
        st.error("Error: Target column 'maintenance_required' not found in CSV.")
    else:
        st.sidebar.success("File uploaded successfully!")
        
        if st.sidebar.button("🚀 Train Models"):
            with st.spinner("Processing data and training models..."):
                # 1. Preprocess
                X, y, num_cols, cat_cols = preprocess_data(df)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                preprocessor = get_preprocessing_pipeline(num_cols, cat_cols)
                
                # 2. Train
                trained_pipelines = train_models(X_train, y_train, preprocessor)
                
                # 3. Evaluate
                test_results = {}
                for name, pipeline in trained_pipelines.items():
                    test_results[name] = evaluate_model(pipeline, X_test, y_test)
                
                # 4. Results Display
                st.markdown("---")
                st.header("📈 Model Performance Comparison")
                
                cols = st.columns(len(test_results))
                
                best_model_name = ""
                best_accuracy = 0
                
                for i, (name, metrics) in enumerate(test_results.items()):
                    with cols[i]:
                        st.subheader(f"🔹 {name}")
                        st.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
                        st.metric("F1-Score", f"{metrics['F1-score']:.2f}")
                        
                        if metrics['Accuracy'] > best_accuracy:
                            best_accuracy = metrics['Accuracy']
                            best_model_name = name
                        
                        # Confusion Matrix Plot
                        fig, ax = plt.subplots(figsize=(4, 3))
                        sns.heatmap(metrics['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_title(f"Conf. Matrix: {name}")
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        st.pyplot(fig)

                st.markdown("---")
                st.info(f"🏆 **Best Model: {best_model_name}** with **{best_accuracy:.2%}% Accuracy**")
                
                # Feature Importance for Decision Tree
                if "Decision Tree" in trained_pipelines:
                    st.subheader("🌲 Feature Importance (Decision Tree)")
                    # We use the raw feature names from X for mapping back
                    # Note: In production, we'd handle OneHot names via preprocessor.get_feature_names_out()
                    importances = get_feature_importance(trained_pipelines["Decision Tree"], X.columns)
                    if importances:
                        imp_df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
                        fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
                        sns.barplot(x='Importance', y='Feature', data=imp_df.head(10), palette='viridis', ax=ax_imp)
                        st.pyplot(fig_imp)

else:
    st.info("👋 Please upload a CSV file to begin. The file should contain columns like mileage, engine_hours, fault_code_count, etc.")
    
    st.markdown("""
        ### Required Format:
        - `vehicle_id` (ID)
        - `mileage` (Integer)
        - `engine_hours` (Float)
        - `fault_code_count` (Integer)
        - `last_service_days` (Integer)
        - `temperature_avg` (Float)
        - `maintenance_required` (Target: 0 or 1)
    """)
