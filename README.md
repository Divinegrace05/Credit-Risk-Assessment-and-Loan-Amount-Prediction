# Credit Risk Assessment and Loan Prediction

![Credit Risk Assessment and Loan Prediction](Risk.jpeg)

## Overview

This project involves analyzing a loan dataset to predict credit risk and loan amounts. The dataset contains key information about borrowers, including age, income, home ownership, employment length, loan intent, loan grade, credit history, and more. 

The project includes:
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Building and evaluating machine learning models
- Model interpretability using SHAP and LIME

### Objectives:
1. **Credit Risk Classification**: To predict whether a loan applicant will default or not based on their profile.
2. **Loan Amount Prediction**: To predict suitable loan amounts for applicants who are likely to not default.

## Business Understanding

### Stakeholders
- **Financial Institutions** (e.g., Banks, Credit Unions): Responsible for managing lending and risk.
- **Loan Applicants**: Borrowers seeking loans for various purposes.
- **Credit Bureaus**: Agencies managing credit reports.
- **Investors & Shareholders**: Those with stakes in financial institutions.
- **Regulatory Bodies**: Ensuring compliance in lending practices.
- **Loan Officers**: Individuals assessing loan eligibility.

### Key Questions:
1. Which borrower profiles are most likely to default on their loans?
2. What factors most strongly influence loan approvals?
3. How can institutions optimize loan offerings for various demographics?
4. How effective are machine learning models in predicting loan amounts and credit risk?

## Data Understanding

### Dataset Source
The dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset?resource=download), consisting of 32,581 observations across 12 variables, including:

- `person_age`: Age of the borrower
- `person_income`: Borrower’s annual income
- `person_home_ownership`: Type of home ownership
- `person_emp_length`: Employment length in years
- `loan_intent`: Purpose of the loan
- `loan_grade`: Loan grade based on creditworthiness (A-G)
- `loan_amnt`: Loan amount
- `loan_int_rate`: Interest rate on the loan
- `loan_status`: Loan default status (0 = non-default, 1 = default)
- `loan_percent_income`: Loan amount as a percentage of income
- `cb_person_cred_hist_length`: Borrower's credit history length in years
- `cb_person_default_on_file`: Previous default status

## Exploratory Data Analysis (EDA)

1. **Loan Intent**:
   - Educational loans make up 19.86% of the total loans, followed by medical and personal reasons.
   - Home improvement loans constitute only 11.08%.

   ![Loan Intent of Borrowers](Images/Loan_Intent_of_Borrowers.png)

2. **Default on File Distribution**:
   - 82% of borrowers have defaulted before, while 18% have clean records.

   ![Default on file distribution](Images/default_on_file_distribution.png)

3. **Loan Intent by Age Group**:
   - Borrowers aged 21-30 mainly take loans for education, while older borrowers (51-60) focus on personal and medical reasons.

   ![Loan Intent by Age Group](Images/loan_intent_by_age_group)

4. **Default Status by Age Group**:
   - Younger borrowers (21-30) tend to default more often.

   ![Default Status by Age Group](Images/default_status_by_age_group)

## Data Preprocessing

Data preprocessing involved:
- **Handling missing values**
- **Encoding categorical variables** using techniques other than one-hot encoding (avoiding dummy variables)
- **Standardizing numerical features** for model performance improvement
- **Splitting data** into train and test sets

## Modeling

### Credit Risk Classification (Binary Classification)

Six machine learning models were evaluated:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Gradient Boosting
6. XGBoost (final model)

#### Model Performance:
| Model              | Accuracy | Precision | Recall  | F1 Score |
|--------------------|----------|-----------|---------|----------|
| Logistic Regression| 0.850    | 0.750     | 0.474   | 0.581    |
| Decision Tree      | 0.882    | 0.714     | 0.766   | 0.739    |
| Random Forest      | 0.934    | 0.964     | 0.725   | 0.828    |
| KNN                | 0.886    | 0.813     | 0.625   | 0.706    |
| Gradient Boosting  | 0.926    | 0.944     | 0.705   | 0.808    |
| **XGBClassifier**  | **0.938**| **0.957** | **0.752**| **0.842**|

### Loan Amount Prediction (Regression)

Three models were used to predict loan amounts:
1. Linear Regression (baseline)
2. XGBoost (best performer)
3. Artificial Neural Networks (ANN)

#### Regression Performance:
| Model              | RMSE    | MSE        | MAE     | R²        |
|--------------------|---------|------------|---------|-----------|
| Linear Regression  | 3684.03 | 1.357e+07  | 2417.26 | 0.654     |
| ANN                | 630.32  | 3.973e+05  | 388.39  | 0.990     |
| **XGBRegressor**   | **498.29**| **227.17** | **751.80**| **0.994**|

## Model Interpretation

### XGBClassifier Interpretation:
- **LIME** was used for local interpretability, showing how different features impacted the probability of loan default.
- **Top Features**: `loan_grade`, `person_income`, `cb_person_cred_hist_length`, `person_emp_length`.

### XGBRegressor Interpretation:
- **SHAP** summary plot showed that `loan_percent_income` and `person_income` were the most impactful features.
- Higher values for these features increased the loan amount prediction, while lower values decreased the predicted amount.

## Key Results

- **XGBClassifier** achieved high precision (95.7%) in identifying non-default loans.
- **XGBRegressor** produced highly accurate loan amount predictions with R² of 0.994, RMSE of 498.29.
- **Top features** influencing predictions were borrower income, credit history length, and loan grade.

## Conclusion

The project demonstrates how machine learning models, especially XGBoost, can significantly improve credit risk assessment and loan amount predictions. Financial institutions can leverage these models to minimize risk, make informed lending decisions, and provide personalized loan offerings.

### Recommendations
1. **Tailored Loan Products**: Customize loans for different borrower segments based on the model’s insights.
2. **Use ML for Risk Assessment**: Implement machine learning models like XGBoost to improve the accuracy of loan approval processes.
3. **Financial Literacy Programs**: Improve borrowers' financial health by offering educational initiatives.
4. **Continuous Model Monitoring**: Regularly update models as new data becomes available to maintain predictive performance.


Instructions to Run the Project
Prerequisites
Python 3.7+
Libraries: pandas, scikit-learn, xgboost, shap, lime, streamlit
Install dependencies using:

bash
Copy code
pip install -r requirements.txt
Running the Streamlit App
To start the Streamlit app, run the following command:

bash
Copy code
streamlit run app.py
Once the app is running, navigate to the provided local URL in your browser to interact with the loan prediction models.

.
├── app.py                  # Streamlit application
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── data/                   # Dataset files
├── notebooks/              # Jupyter Notebooks with analysis and model development
├── models/                 # Saved model files
└── visuals/                # Visual outputs and graphs




# Credit-Risk-Assessment-with-Loan-Amount-Prediction
![creditRiskLoanPredApp - Google Chrome 04_10_2024 02_24_01](https://github.com/user-attachments/assets/18110a12-ba21-4935-9990-1076ee928c5a)
![creditRiskLoanPredApp - Google Chrome 04_10_2024 02_24_15](https://github.com/user-attachments/assets/32096529-563e-4ce3-a1bd-b36e357c9ab7)
![creditRiskLoanPredApp - Google Chrome 04_10_2024 15_17_36](https://github.com/user-attachments/assets/7a5290c9-44b6-4bcc-a815-e8ed92b3e92a)


outputs
![creditRiskLoanPredApp - Google Chrome 04_10_2024 14_52_16](https://github.com/user-attachments/assets/18f9907b-171d-4dca-9e18-f52ede0e0976)
![creditRiskLoanPredApp - Google Chrome 04_10_2024 14_52_32](https://github.com/user-attachments/assets/e8b06b62-cdba-48b3-9478-f909b8c66074)
