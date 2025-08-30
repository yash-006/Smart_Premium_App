# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from data_prep import FeatureEngineeringTransformer

# -------------------------
# Must be the first Streamlit command
# -------------------------
st.set_page_config(page_title="Smart Premium Predictor", layout="wide")

# -------------------------
# Load the trained pipeline
# -------------------------
@st.cache_resource
def load_model():
    model = joblib.load("../models/final_premium_xgb_pipeline.pkl")
    return model

model = load_model()

# -------------------------
# Streamlit App
# -------------------------
st.title("💡 Smart Premium Prediction App")
st.markdown("Predict health insurance premiums using Machine Learning 🚀")

# Sidebar Navigation
option = st.sidebar.radio("Choose Mode", ["Single Prediction", "Batch Prediction"])

# -------------------------
# Single Prediction Section
# -------------------------
if option == "Single Prediction":
    st.subheader("🔹 Single Prediction")

    with st.form("single_prediction_form"):
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
        occupation = st.text_input("Occupation")
        smoking_status = st.selectbox("Smoking Status", ["Smoker", "Non-Smoker"])
        exercise_frequency = st.selectbox("Exercise Frequency", ["None", "1-2 times/week", "3-5 times/week", "Daily"])
        location = st.text_input("Location")
        property_type = st.selectbox("Property Type", ["Apartment", "House", "Condo"])
        policy_type = st.selectbox("Policy Type", ["Health", "Life", "Auto", "Home"])
        income = st.number_input("Annual Income", min_value=10000, max_value=10000000, step=1000, value=500000)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
        policy_date = st.date_input("Policy Start Date")
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)
        health_score = st.slider("Health Score", min_value=0, max_value=100, value=50)
        previous_claims = st.number_input("Previous Claims", min_value=0, max_value=50, step=1)
        vehicle_age = st.number_input("Vehicle Age (in years)", min_value=0, max_value=50, step=1)
        insurance_duration = st.number_input("Insurance Duration (in years)", min_value=0, max_value=50, step=1)
        feedback = st.text_area("Customer Feedback (optional)")

        submitted = st.form_submit_button("Predict Premium")

    if submitted:
        try:
            # ✅ Build dataframe for model
            input_df = pd.DataFrame([{
                "Age": age,
                "Gender": gender,
                "Marital Status": marital_status,
                "Education Level": education,
                "Occupation": occupation,
                "Smoking Status": smoking_status,
                "Exercise Frequency": exercise_frequency,
                "Location": location,
                "Property Type": property_type,
                "Policy Type": policy_type,
                "Annual Income": income,
                "Credit Score": credit_score,
                "Policy Start Date": policy_date,
                "Number of Dependents": dependents,
                "Health Score": health_score,
                "Previous Claims": previous_claims,
                "Vehicle Age": vehicle_age,
                "Insurance Duration": insurance_duration,
                "Customer Feedback": feedback,
            }])

            # ✅ Use input_df (NOT input_data)
            prediction = model.predict(input_df)[0]
            st.success(f"💰 Predicted Premium Amount: ₹{prediction:,.2f}")

        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")

# ==========================
# Batch Prediction Section
# ==========================
st.header("📂 Batch Premium Prediction")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    try:
       # Predict (direct, no inverse transform needed now)
        predictions = model.predict(df)

        df["Predicted Premium Amount"] = predictions

        # Show preview
        st.dataframe(df)

        # Allow download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"❌ Error making batch prediction: {e}")


# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("🔐 Built with XGBoost, Streamlit, and love ❤️ | © 2025 Insurance Premium Predictor")
