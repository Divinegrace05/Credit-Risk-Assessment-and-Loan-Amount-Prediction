import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings

# Load the saved models, feature names, and scalers
with open("xgb_classifier.pkl", "rb") as file:
    xgb_classifier = pickle.load(file)

with open("xgb_regressor.pkl", "rb") as file:
    xgb_regressor = pickle.load(file)

# Load classification-specific feature names and scaler
with open("classification_features.pkl", "rb") as f:
    classification_feature_names = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    classification_scaler = pickle.load(f)

# Load regression-specific feature names and scaler
with open("regression_features.pkl", "rb") as f:
    regression_feature_names = pickle.load(f)

with open("scaler_reg.pkl", "rb") as f:
    regression_scaler = pickle.load(f)

# Function to preprocess input data for classification
def preprocess_classification_input(data):
    df = pd.DataFrame([data])
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    for col in classification_feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[classification_feature_names]
    df_scaled = classification_scaler.transform(df_encoded)
    return df_scaled

# Function to preprocess input data for regression
def preprocess_regression_input(data):
    df = pd.DataFrame([data])
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    for col in regression_feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[regression_feature_names]
    df_scaled = regression_scaler.transform(df_encoded)
    return df_scaled

# Streamlit app
st.title('Credit Risk Assessment and Loan Prediction')

# Input fields
st.header('Borrower Information')
person_age = st.number_input('Age', min_value=18, max_value=100, value=30)
person_income = st.number_input('Annual Income', min_value=0, value=50000)
person_home_ownership = st.selectbox('Home Ownership', ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
person_emp_length = st.number_input('Employment Length (years)', min_value=0, max_value=50, value=5)
loan_intent = st.selectbox('Loan Intent', ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
loan_grade = st.selectbox('Loan Grade', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
loan_int_rate = st.slider('Interest Rate', min_value=1.0, max_value=25.0, value=10.0)
loan_percent_income = st.slider('Loan Percent Income', min_value=0.0, max_value=1.0, value=0.1)
cb_person_cred_hist_length = st.number_input('Credit History Length (years)', min_value=0, max_value=50, value=5)
cb_person_default_on_file = st.selectbox('Previous Default', ['Y', 'N'])

# Initialize session state
if 'credit_risk_prediction' not in st.session_state:
    st.session_state.credit_risk_prediction = None

# Button to make credit risk prediction
if st.button('Predict Credit Risk'):
    # Prepare input data for classification
    input_data = {
        'person_age': person_age,
        'person_income': person_income,
        'person_home_ownership': person_home_ownership,
        'person_emp_length': person_emp_length,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'cb_person_default_on_file': cb_person_default_on_file
    }
    
    # Preprocess input data for classification
    processed_input_classification = preprocess_classification_input(input_data)
    
    # Make credit risk prediction
    st.session_state.credit_risk_prediction = xgb_classifier.predict(processed_input_classification)[0]
    
    # Display results based on credit risk prediction
    if st.session_state.credit_risk_prediction == 0:
        st.header('Good News! You are eligible for a loan.')
    
        # Button to predict loan amount (only shown if eligible)
        if st.button('Predict Loan Amount'):
    
            # Preprocess input data for regression
            processed_input_regression = preprocess_regression_input(input_data)
            
            # Predict loan amount
            loan_amount_prediction = xgb_regressor.predict(processed_input_regression)[0]
            
            st.write(f"Predicted Loan Amount: ${loan_amount_prediction:.2f}")

    else:
        st.header('You are not eligible for a loan.')
