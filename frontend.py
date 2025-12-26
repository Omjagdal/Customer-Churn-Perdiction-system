import streamlit as st
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open("customer_churn.pkl", "rb"))

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("💼 Customer Churn Prediction System")
st.write("Fill in the details below to predict whether a customer will churn.")

# ===================== INPUT FIELDS ===================== #
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
    monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=70.0)
    total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=900.0)

# Convert user input into a dictionary
input_data = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}

# ===================== PREDICTION ===================== #
if st.button("🔍 Predict Churn"):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]

    if prediction == 1:
        st.error("⚠️ The customer is LIKELY to churn. Take preventive actions.")
    else:
        st.success("✅ The customer is NOT likely to churn.")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit + XGBoost")
