# Credit Risk Assessment and Loan Amount Prediction
![Credit Risk](https://github.com/user-attachments/assets/bffc185a-2ba8-414e-a6f6-0ce799b40b19)


## Project Summary
This project addresses critical challenges in financial lending by analyzing a loan dataset to predict credit risk and loan amounts. The dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset?resource=download), contains 32,581 observations across 12 variables, including borrower demographics, financial indicators, and loan characteristics.

Data preparation included handling missing values, removing duplicates, and addressing outliers in income and employment length. We used pandas for data manipulation and scikit-learn for preprocessing, applying StandardScaler for feature scaling and LabelEncoder for categorical variables.

For credit risk prediction, we tested multiple models such as Logistic Regression, Decision Trees, Random Forest, K-Nearest Neighbors, Gradient Boosting, and XGBoost. XGBoost proved most effective, achieving 93.8% accuracy and an F1 score of 0.842. Loan amount prediction employed Linear Regression, XGBoost, and Artificial Neural Networks, with XGBoost again leading with an R² of 0.99 and an RMSE of 498.28.

Model performance was assessed using an 80-20 train-test split, with metrics such as accuracy, precision, and F1 score for classification, and RMSE, MSE, and R² for regression. SHAP and LIME were applied to enhance interpretability, identifying income, loan-to-income ratio, and loan intent as key predictors.

The project was deployed using Streamlit, offering an interactive interface for real-time credit risk and loan amount predictions. This deployment enhances accessibility, providing valuable tools for lenders to make informed decisions.

## Business Understanding
With the growing significance of consumer credit and lending, accurate and efficient credit risk assessment is crucial for both financial institutions and customers. These assessments safeguard banks' financial health, increase credit volume, and promote responsible lending, ensuring economic stability and loans for reliable borrowers. Misjudging a borrower’s risk can lead to losses for institutions, while inaccurate loan amounts may either overburden the customer or limit their potential.

Accurate loan amount prediction is essential for financial institutions to avoid excessive risk. Credit risk classification segments borrowers based on their creditworthiness, influencing loan approvals and interest rates. Loan amount prediction uses factors like income, employment stability, and credit history to estimate a suitable loan, reducing the risk of default.

By applying machine learning, lenders can balance credit access with maintaining a healthy loan portfolio, benefiting both borrowers and financial institutions.


### Stakeholders
- Financial Institutions (e.g., Banks, Credit Unions): Responsible for managing lending and risk.
- Loan Applicants: Borrowers seeking loans for various purposes.
- Credit Bureaus: Agencies managing credit reports.
- Investors & Shareholders: Those with stakes in financial institutions.
- Regulatory Bodies: Ensuring compliance in lending practices.
- Loan Officers: Individuals assessing loan eligibility.

### Key Questions:
1. Which borrower profiles are most likely to default on their loans?
2. What factors most strongly influence loan approvals?
3. How can institutions optimize loan offerings for various demographics?
4. How effective are machine learning models in predicting loan amounts and credit risk?

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

   ![Loan Intent by Age Group](Images/loan_intent_by_age_group.png)

4. **Default Status by Age Group**:
   - Younger borrowers (21-30) tend to default more often.

   ![Default Status by Age Group](Images/default_status_by_age_group.png)
   

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
![Lime Viz](Images/LimeViz.png)
![e8498eec-54a2-4b76-a572-c4118c50e35e](https://github.com/user-attachments/assets/79fc1c01-7ffe-4eaf-b700-2b6d20395b1f)
- **LIME** was used for local interpretability, showing how different features impacted the probability of loan default.
- **Top Features**: `loan_intent`, `loan_grade`, `person_income`, `person_emp_length`.

### XGBRegressor Interpretation:
![0ecd2439-b8d8-47d6-b7a1-55f1ab3f77cb](https://github.com/user-attachments/assets/d5265862-89a5-4972-a881-d89b9163dc61)
![fdc49a8d-6319-433c-a644-1e695343d419](https://github.com/user-attachments/assets/b9878011-f3b0-45c2-8fc5-74193049d1ef)
- **SHAP** summary plot showed that `loan_percent_income` and `person_income` were the most impactful features.
- Higher values for these features increased the loan amount prediction, while lower values decreased the predicted amount.

## Key Results

- **XGBClassifier** achieved high precision (93.8%) in identifying non-default loans.
- **XGBRegressor** produced highly accurate loan amount predictions with R² of 0.99, RMSE of 498.29.
- **Top features** influencing predictions were borrower income, credit history length, and loan grade.

## Deployment
[Credit Risk and Loan Amount Prediction App](https://loanamountprediction.streamlit.app/)

## Key Findings

### 1. Borrower Profiles and Loan Intentions
- Many loans were taken for **educational purposes** and **home improvement**.
- A significant portion of borrowers are **renters**, not homeowners.
- **Recommendation**: Financial institutions should design loan products that cater to renters and young professionals, offering flexible terms and repayment options.

### 2. Default History and Risk Mitigation
- A notable number of borrowers had **default histories**, emphasizing the need for robust risk assessment models.
- **Recommendation**: Implement more sophisticated machine learning models and **financial literacy programs** to help borrowers manage credit and reduce default risks.

### 3. Demographic Insights and Borrowing Patterns
- **Younger borrowers** (ages 21-30) mainly seek loans for education.
- **Older borrowers** tend to request loans for **home improvement** or **debt consolidation**.
- **Recommendation**: Lenders should develop **demographic-specific financial products** with favorable terms for educational loans and flexible options for older borrowers.

### 4. Model Performance
- **XGBoost** models outperformed other algorithms for both:
  - **Credit Risk Classification**: Achieved 93% accuracy and 95% precision.
  - **Loan Amount Prediction**: Achieved an RMSE of 498 and an R² score of 0.99.
- **Hyperparameter tuning** did not significantly improve performance, suggesting XGBoost performs well with default settings.

### 5. Feature Importance
- Key features for loan outcomes include:
  - **Borrower income**
  - **Loan grade**
  - **Loan percent income**
- **Recommendation**: Financial institutions can refine their credit policies using these insights to better understand borrower behavior and risk.

## Recommendations for Financial Institutions
1. **Custom Loan Products**: Develop loan offerings tailored to specific borrower segments, such as young professionals and education seekers.
2. **Enhance Risk Assessment**: Incorporate machine learning models like XGBoost to predict credit risk with high accuracy.
3. **Financial Literacy Programs**: Educate borrowers to reduce default rates and promote healthier financial habits.
4. **Continuous Model Monitoring**: Regularly update models with new borrower data to maintain accuracy and relevance.
5. **Advanced Techniques**: Explore ensemble learning methods to improve credit risk prediction models.

This project demonstrates the power of machine learning in revolutionizing credit risk assessment and loan prediction. By adopting these techniques, financial institutions can make more informed decisions, reduce default rates, and better serve diverse borrower profiles. Continued investment in machine learning and AI technologies will be key to optimizing financial services and driving sustainable growth.


## Getting Started

### Prerequisites
- Python 3.7+
- Libraries: pandas, scikit-learn, xgboost, shap, lime, streamlit

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/Divinegrace05/Credit-Risk-Assessment-and-Loan-Amount-Prediction
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Streamlit App

To run the Streamlit app locally:

```
streamlit run app.py
```

Navigate to the provided local URL in your browser to interact with the loan prediction models.

## Thank You
