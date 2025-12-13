# CUSTOMER-CHURN-PREDICTION-
A MACHINE LEARNING PROJECT TO PREDICT CUSTOMER CHURN INTEGRATING WEB API (FASTAPI)
CUSTOMER CHURN PREDICTION SYSTEM (ML MODEL + FASTAPI WEB API)

Dataset source: https://www.kaggle.com/datasets/abdallahwagih/telco-customer-churn?resource=download&select=Telco_customer_churn.xlsx

This project provides an end-to-end Machine Learning solution for predicting customer churn. It includes:

Data preprocessing & feature engineering

Model training & evaluation

SHAP model interpretability

A production-ready FastAPI Web API for real-time predictions

Saved model artifacts (.pkl files) and JSON feature mappings

THIS PROJECT USES A LOGISTIC REGRESSION CLASSIFIER

Why Logistic Regression?

Easy to interpret

Works well with linearly separable data

Efficient on large datasets

Stable, reliable probability outputs

KEY STEPS IN THE ML PIPELINE

Handling missing data; the missing values were the churn reason which was excluded from model training

Label encoding & One-Hot Encoding

Export of correct column order via model_columns.json

Fitting Logistic Regression on the processed dataset

Saving all artifacts:

churn_model.pkl

scaler.pkl

model_columns.json

Standardizing numeric features

Preserving feature order during inference

Training a Logistic Regression classifier

FEATURES (WEB UI)

Users can input customer data through a Bootstrap-powered HTML form.
The result page shows:

Churn / Not Churn prediction

Colored text (red or green)

Probability of churn

REST API

Send JSON to /predict and receive:

Predicted class (0 = Not Churn, 1 = Churn)

Churn probability

Flask-Based Backend:

Serves HTML templates

Handles form POST requests

Runs model inference in real-time

📁 Project Structure
project/
│
├── app.py
├── churn_model.pkl
├── scaler.pkl
├── model_columns.json
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── static/
    ├── style.css
    └── img/
        └── churn.jpg
<img width="321" height="479" alt="Screenshot 2025-11-28 201307" src="https://github.com/user-attachments/assets/4cb8af94-2a21-4338-b865-ab369aec0f72" />


INSTALLATION & SETUP
1. INSTALL DEPENDENCIES
pip install flask pandas scikit-learn joblib

2. ENSURE YOUR ML ARTIFACTS ARE PRESENT

churn_model.pkl

scaler.pkl

model_columns.json

3. RUN THE FLASK APP
python app.py

4. OPEN THE APPLICATION
http://127.0.0.1:5000/

API USAGE
ENDPOINT
POST /predict

EXAMPLE JSON REQUEST
{
  "Tenure_Months": 18,
  "Monthly_Charges": 72.3,
  "Churn_Score": 41,
  "CLTV": 3980,
  "Churn_Value": 0
}

EXAMPLE JSON RESPONSE (LOGISTIC REGRESSION OUTPUT)
{
  "prediction": 1,
  "probability": 0.8124
}

FRONTEND PAGES
index.html

Customer input form

Bootstrap & custom CSS

Displays project image (churn.jpg)

result.html

Shows prediction

Red = Churn

Green = Not Churn

Displays churn probability

"Back to Home" button

IMAGE ASSET

Place your image inside:

static/img/churn.jpg


Displayed at the top of the homepage.

MODEL NOTES (LOGISTIC REGRESSION INSIGHTS)

Logistic Regression:

Outputs probabilities using the sigmoid function

Ideal for churn classification due to:

Binary nature (0 or 1)

Well-calibrated probability outputs

Coefficients help interpret feature impact on churn

Works best when:

Features are scaled

Data is not highly nonlinear

Multicollinearity is handled

CONTRIBUTIONS

You can contribute by:

Improving the model

Adding new features

Enhancing UI

Writing tests

Pull requests are welcome.

ACKNOWLEDGEMENTS

Flask

Scikit-learn

Bootstrap CSS

Logistic Regression theory

Pandas + joblib for ML integration

REFERENCES

Ouko, Allan. Customer Churn Prediction.
https://github.com/AllanOuko/customer-churn-prediction-application/tree/main

APP INPUT USER INTERFACE

<img width="971" height="711" alt="Screenshot 2025-11-28 190536" src="https://github.com/user-attachments/assets/3eacb821-d24a-4226-99b8-e107156d7c91" />

APP RESULT USER INTERFACE

<img width="1886" height="964" alt="Screenshot 2025-11-28 190606" src="https://github.com/user-attachments/assets/0369badf-179b-4812-9ae0-645ce4d0eb8c" />



