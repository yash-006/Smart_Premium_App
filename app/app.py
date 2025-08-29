# app.py

# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
import os

# ==========================
# MUST be first Streamlit call
# ==========================
st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="💰",
    layout="centered"
)

# ====================================
# 0. Path Setup for Custom Transformer
# ====================================
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data_prep import FeatureEngineeringTransformer  # Custom feature transformer

# ==========================
# 1. Load Trained Pipeline
# ==========================
@st.cache_data
def load_pipeline():
    try:
        pipeline = joblib.load("../models/final_premium_xgb_pipeline.pkl")
        return pipeline
    except Exception as e:
        st.error(f"❌ Error loading pipeline: {e}")
        return None

pipeline = load_pipeline()

# ==========================
# 2. App Title & Intro
# ==========================
st.title("💰 Insurance Premium Predictor")
st.write("Predict insurance premiums using our trained **XGBoost model**. "
         "You can upload a CSV file for batch predictions or enter details manually for a single prediction.")
st.markdown("---")

# ==========================
# 3. CSV Upload Option
# ==========================
st.header("📂 Upload CSV for Batch Predictions")
st.caption("Upload a CSV file with customer details to predict insurance premiums for multiple entries at once.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file and pipeline:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("🔍 Data Preview:")
        st.dataframe(data.head())

        if st.button("🚀 Predict Premiums for CSV"):
            predictions = pipeline.predict(data)
            data["Predicted Premium"] = np.expm1(predictions)  # inverse log1p

            st.success("✅ Predictions Complete!")
            st.dataframe(data)

            # Download predictions
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv,
                file_name="predicted_premiums.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"❌ Error processing CSV: {e}")

st.markdown("---")

# ==========================
# 4. Manual Input Option
# ==========================
st.header("🧑‍💻 Predict Premium for Single Input")
st.caption("Enter customer details below to get an instant premium prediction.")

if pipeline:
    with st.form("single_input_form"):
        st.write("### Enter Customer Details:")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        annual_income = st.number_input("Annual Income (₹)", min_value=1000, value=50000, step=1000)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
        policy_start_date = st.date_input("Policy Start Date")
        customer_feedback = st.text_area("Customer Feedback (optional)")

        submitted = st.form_submit_button("🔮 Predict Premium")

        if submitted:
            try:
                input_df = pd.DataFrame([{
                    "Age": age,
                    "Gender": gender,
                    "Marital Status": marital_status,
                    "Annual Income": annual_income,
                    "Credit Score": credit_score,
                    "Policy Start Date": policy_start_date,
                    "Customer Feedback": customer_feedback
                }])

                pred = pipeline.predict(input_df)
                premium = np.expm1(pred[0])  # inverse log1p

                st.success(f"💵 Predicted Premium: **₹{premium:,.2f}**")
            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")
else:
    st.warning("⚠️ Model pipeline not loaded. Please check model path.")

# ==========================
# 5. Footer
# ==========================
st.markdown("---")
st.caption("🔐 Built with XGBoost, Streamlit, and love ❤️ | © 2025 Insurance Premium Predictor")
