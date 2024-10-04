import streamlit as st 
import pickle
import numpy as np
import pandas as pd

# Load the saved models
with open("xgb_classifier.pkl", "rb") as file:
    xgb_classifier = pickle.load(file)

with open("xgb_regressor.pkl", "rb") as file:
    xgb_regressor = pickle.load(file)

# Define the feature names for classification
classification_feature_names = ['person_age', 'person_income', 'person_home_ownership', 
                                 'person_emp_length', 'loan_intent', 'loan_grade', 
                                 'loan_int_rate', 'loan_percent_income', 
                                 'cb_person_default_on_file', 'cb_person_cred_hist_length']

# Define the feature names for regression (excluding loan_int_rate)
regression_feature_names = ['person_age', 'person_income', 'person_home_ownership', 
                            'person_emp_length', 'loan_intent', 'loan_grade', 
                            'loan_percent_income', 'cb_person_default_on_file', 
                            'cb_person_cred_hist_length']

# Function to preprocess input data for classification
def preprocess_input_classification(data):
    home_ownership_map = {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'OTHER': 3}
    loan_intent_map = {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 
                       'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5}
    loan_grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    default_history_map = {'Y': 1, 'N': 0}

    data['person_home_ownership'] = home_ownership_map[data['person_home_ownership']]
    data['loan_intent'] = loan_intent_map[data['loan_intent']]
    data['loan_grade'] = loan_grade_map[data['loan_grade']]
    data['cb_person_default_on_file'] = default_history_map[data['cb_person_default_on_file']]
    
    return data

# Function to preprocess input data for regression
def preprocess_input_regression(data):
    print(f"Input data in preprocess_input_regression: {data}")  # Debug log
    
    home_ownership_map = {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'OTHER': 3}
    loan_intent_map = {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5}
    loan_grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    default_history_map = {'Y': 1, 'N': 0}

    # Remove the loan_int_rate since it's not needed for regression
    data.pop('loan_int_rate', None)

    # Apply mappings with error handling
    try:
        data['person_home_ownership'] = home_ownership_map[data['person_home_ownership']]
        data['loan_intent'] = loan_intent_map[data['loan_intent']]
        data['loan_grade'] = loan_grade_map[data['loan_grade']]
        data['cb_person_default_on_file'] = default_history_map[data['cb_person_default_on_file']]
    except KeyError as e:
        print(f"KeyError in preprocess_input_regression: {e}")
        print(f"Current data state: {data}")
        raise

    print(f"Processed data in preprocess_input_regression: {data}")  # Debug log
    return data

# Streamlit app
st.title('Credit Risk Assessment and Loan Amount Prediction')

# Input fields
st.header('Borrower Information')
age = st.number_input('Age', min_value=18, max_value=100, value=26)
income = st.number_input('Annual Income', min_value=0, value=90000)
home_ownership = st.selectbox('Home Ownership', ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
emp_length = st.number_input('Employment Length (years)', min_value=0, max_value=50, value=2)
loan_intent = st.selectbox('Loan Intent', ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
loan_grade = st.selectbox('Loan Grade', ['G', 'F', 'E', 'D', 'C', 'B', 'A'])
loan_int_rate = st.slider('Loan Interest Rate', 0.0, 30.0, 11.14)  # Used for classification only
loan_percent_income = st.slider('Loan Percent Income', 0.0, 1.0, 0.60)
default_history = st.selectbox('Default History', ['Y', 'N'])
credit_history_length = st.number_input('Credit History Length (years)', min_value=0, max_value=50, value=5)

# Create a dictionary with the input data
input_data = {
    'person_age': age,
    'person_income': income,
    'person_home_ownership': home_ownership,
    'person_emp_length': emp_length,
    'loan_intent': loan_intent,
    'loan_grade': loan_grade,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_percent_income,
    'cb_person_default_on_file': default_history,
    'cb_person_cred_hist_length': credit_history_length
}

# Button to predict credit risk
if st.button('Predict'):
    print(f"Original input data: {input_data}")  # Debug log
    
    # Create separate copies for classification and regression
    input_data_classification = input_data.copy()
    input_data_regression = input_data.copy()
    
    # Preprocess input data for classification
    processed_data_classification = preprocess_input_classification(input_data_classification)
    print(f"Processed classification data: {processed_data_classification}")  # Debug log
    
    # Preprocess input data for regression
    processed_data_regression = preprocess_input_regression(input_data_regression)
    print(f"Processed regression data: {processed_data_regression}")  # Debug log
 
    # Convert to numpy array for classification
    input_array_classification = np.array([processed_data_classification[feature] for feature in classification_feature_names]).reshape(1, -1)
    
    # Make prediction for credit risk
    prediction = xgb_classifier.predict(input_array_classification)[0]
    
    if prediction == 1:
        st.error("We're sorry, but you are not eligible for a loan at this time.")
        st.image("Rejected.gif")

    else:
        # Load the scaler for regression
        with open("scaler_reg.pkl", "rb") as file:
            scaler_reg = pickle.load(file)       
        
        # Convert processed data to DataFrame for scaling
        regression_input_df = pd.DataFrame([processed_data_regression], columns=regression_feature_names)
        
        # Scale the regression input
        scaled_regression_input = scaler_reg.transform(regression_input_df)
        
        # Make the loan amount prediction using the XGBRegressor
        loan_amount = xgb_regressor.predict(scaled_regression_input)
        
        # Log the loan amount prediction output
        st.success(f"Good News! You're eligible for a loan amount of {loan_amount[0]:,.2f}")
        st.image("Approved.gif")
 