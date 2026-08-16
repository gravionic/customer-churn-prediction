# Customer Churn Prediction using Machine Learning

## Project Overview

Customer churn is one of the biggest challenges for subscription-based businesses. This project predicts whether a customer is likely to leave the company and explains each prediction using SHAP (Explainable AI). A Power BI dashboard provides business insights, while a Streamlit web application enables interactive predictions.

---

## Business Problem

Retaining existing customers is significantly more cost-effective than acquiring new ones. This project helps businesses:

- Predict customers at risk of churn
- Understand the factors driving churn
- Support data-driven customer retention strategies

---
## 📁 Project Structure

customer-churn-prediction/
│
├── app/
│   └── app.py
│
├── assets/
│   └── churn_banner.png
│
├── data/
│   ├── raw/
│   └── processed/
│
├── images/
│   ├── model-comparison.png
│   ├── powerbi-dashboard.png
│   ├── streamlit-app-01.png
│   └── streamlit-app-02.png
│
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│   └── 01.ipynb
│
└── README.md
---

## Dataset

- **Source:** Telco Customer Churn Dataset from kaggle

---
## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Streamlit
- Power BI
- Matplotlib

---

# Machine Learning Pipeline

- Data Cleaning
- Feature Engineering
- One-Hot Encoding
- Feature Scaling
- Model Training
- Hyperparameter Tuning
- Model Evaluation
- SHAP Explainability
- Model Deployment using Streamlit

---

# Models Evaluated

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
![Model Comparison](images/model-comparison.png)

---
# Final Model Selection

Random Forest was selected as the final model because it provided the best balance between:

- ROC-AUC
- Recall
- F1 Score
- Model stability
- Explainability using SHAP

Although XGBoost achieved a slightly higher ROC-AUC, Random Forest produced a better overall balance across evaluation metrics and generated more stable predictions for this business problem. The project prioritizes recall because missing a genuine churner (false positive) can mean losing a customer, while false positive can be followed up with a retention offer.

---

# Model Performance

| Metric | Random Forest |
|---------|--------------:|
| Accuracy | 72.85% |
| Precision | 49.34% |
| Recall | 79.68% |
| F1 Score | 60.94% |
| ROC-AUC | 0.8204 |

---

# Key Findings

The SHAP analysis identified the strongest drivers of customer churn:

- Month-to-month contracts
- No Online Security
- Fiber optic internet service
- Short customer tenure
- No Tech Support
- Electronic check payment method

Customers with longer contracts, longer tenure, and additional security services were considerably less likely to churn.

---

# Streamlit Web Application

The application allows users to:

- Enter customer information
- Predict churn probability
- View customer risk level
- Receive retention recommendations
- Understand predictions through SHAP Explainable AI

![Streamlit App](images/streamlit-app-01.png)
![Streamlit App](images/streamlit-app-02.png)
---

# Power BI Dashboard

The dashboard provides business insights including:

- Customer distribution
- Churn rate
- Revenue analysis
- Contract type analysis
- Payment method analysis
- Customer demographics
- Churn by tenure
- Interactive filters

![Power BI Dashboard](images/powerbi-dashboard.png)

---

# Future Improvements

- Deploy the application to Streamlit Community Cloud
- Add LLM-based retention recommendations
- Integrate customer database support
- Add batch prediction functionality

---

# Author

Mohit