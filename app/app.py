# Customer Churn Prediction Web Application
# Built with Streamlit + Random Forest + SHAP Explainability


# 1. Import Required Libraries
from pathlib import Path
import streamlit as st
import joblib as  jl
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt


Basedir = Path(__file__).resolve().parent.parent
MODEL_DIR = Basedir / "models"
ASSETS_DIR = Basedir / "assets"

# 2. Load the trained model and saved objects
model = jl.load(MODEL_DIR / "model.pkl")
scaler= jl.load(MODEL_DIR / "scaler.pkl")
feature_columns= jl.load(MODEL_DIR / "feature_columns.pkl")

# 3. Page settings
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)
st.image(ASSETS_DIR / "churn_banner.png", width=950)
st.header("Customer Churn Prediction")
st.markdown(
    "<p style='font-size:20px;'>"
    "Predict customer churn using a Machine Learning model and explain each prediction with SHAP Explainable AI."
    "</p>",
    unsafe_allow_html=True
)

# 4. Overview
st.info("""
### Overview

• Predicts whether a customer is likely to churn

• Estimates churn probability

• Explains WHY the prediction was made using SHAP

• Suggests business retention strategies
""")


# 5. Sidebar for user input
st.sidebar.header("Customer Information")


MonthlyCharges=st.sidebar.slider("Monthly Charges",
                                 18,
                                 120,
                                 70)
tenure=st.sidebar.slider("Tenure (Months)", 
                         min_value=0,
                         max_value=72,
                         value=12)
Contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)
InternetService = st.sidebar.selectbox(
    "Internet Service",
    ["Fiber optic", "DSL", "No"]
)
OnlineSecurity = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No", "No Internet Service"]
)

TechSupport = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No", "No Internet Service"]
)


TotalCharges=tenure*MonthlyCharges

# 6. Additional Customer Information
with st.sidebar.expander("Advanced Options"):
    gender=st.selectbox("Gender", ["Male", "Female"])
    Partner=st.selectbox("Partner", ["Yes", "No"])
    Dependents=st.selectbox("Depenedents", ["Yes", "No"])

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )
    multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
    )
    online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
    )
    SeniorCitizen=st.selectbox("Senior Citizen", [0, 1])


    device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
    )

    PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
    )
    PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    )
    

# 7. Predict button
predict = st.sidebar.button("🔍 Predict Churn")

if predict:
    # 8. Preprocess Customer Data
    
    input_dict = {
        "SeniorCitizen": SeniorCitizen,
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,

        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": TechSupport,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
    }

    # 9. Convert input dictionary to DataFrame and preprocess
    input_df = pd.DataFrame([input_dict]) 
    input_df = pd.get_dummies(input_df) # Create dummy variables for categorical features
    for col in feature_columns: # Ensure all expected columns are present
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns] # Reorder columns to match training data
    input_scaled = scaler.transform(input_df) # Scale the input features using the saved scaler

    # 10. Make prediction and calculate probability
    input_scaled_df = pd.DataFrame( 
        input_scaled,
        columns=feature_columns
        )     
    prediction = model.predict(input_scaled)[0] # Make prediction using the trained model
    probability = model.predict_proba(input_scaled)[0][1] # Get the probability of churn
    

    st.markdown("---")

    st.subheader("Prediction Summary")

    # 11. Display prediction results KPIs in three columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability*100:.1f}%"
        )

    with col2:

        if probability < 0.3:
            risk = "Low"

        elif probability < 0.7:
            risk = "Moderate"

        else:
            risk = "High"

        st.metric(
            "Risk Level",
            risk
        )

    with col3:

        st.metric(
            "Revenue at Risk",
            f"${probability * TotalCharges:.0f}"
        )

    st.progress(float(probability))

    # 12. Display risk level with color-coded messages
    if probability < 0.3:
        st.success("🟢 Low Risk")

    elif probability < 0.7:
        st.warning("🟡 Moderate Risk")

    else:
        st.error("🔴 High Risk")

    st.markdown("---")
    # 13. Provide recommended retention actions based on churn probability
    st.subheader("Recommended Retention Actions") 

    if probability >= 0.7:
        st.error("""
    -  Offer long-term contract discount
    -  Provide Online Security free for 3 months
    -  Offer premium Tech Support
    """)

    elif probability >= 0.3:
        st.warning("""
    - Send personalized retention offers
    - Recommend adding Online Security at a discounted price if not added yet
    - Promote Tech Support
    """)

    else:
        st.success("""
    -  Customer appears loyal 
    -  Offer loyalty rewards
    -  Recommend premium plans
    """)

    # 14, Explain individual predictions with SHAP
    st.markdown("---")
    st.subheader("Local SHAP Explanation")
    st.caption(
        "Features pushing this customer towards or away from churn."
    )
    explainer = shap.TreeExplainer(model)
    explanation = explainer(input_scaled_df)

    # 15. Display SHAP waterfall plot for local explanation
    plt.figure(figsize=(10,6))
    shap.plots.waterfall(explanation[:,:,1][0],
    max_display=10,
    show=False
    )
    left, right = st.columns([2,1])

    with left: # Display the SHAP waterfall plot in the left column
        st.pyplot(plt.gcf())

    with right:    # Display key drivers in the right column

        st.subheader("Key Drivers")

        st.write("• Month-to-month contracts increase churn risk.")

        st.write("• Customers without Online Security are more likely to leave.")

        st.write("• Fiber optic users show higher churn.")

        st.write("• Longer tenure reduces churn.")
        plt.close()

        # 16. Display global feature importance using SHAP
        st.markdown("---")
        st.subheader("Global Feature Importance")
        st.caption(
            "Most influential features across the entire dataset."
        )
        fig, ax = plt.subplots(figsize=(10,6))
        shap.plots.bar(
        explanation[:,:,1],
        max_display=10,
        show=False
        )

        with st.expander("View Global Feature Importance"):
            st.pyplot(fig)
        plt.close(fig)
        
