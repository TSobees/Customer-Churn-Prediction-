# Customer-Churn-Prediction-
A machine learning project to predict customer churn integrating Web API FastAPI

#Customer Churn Prediction System (ML Model + FastAPI Web API)
Dataset source: https://www.kaggle.com/datasets/abdallahwagih/telco-customer-churn?resource=download&select=Telco_customer_churn.xlsx

This project provides an end-to-end Machine Learning solution for predicting customer churn. It includes:
Data preprocessing & feature engineering
Model training & evaluation
SHAP model interpretability
A production-ready FastAPI Web API for real-time predictions
Saved model artifacts (.pkl files) and JSON feature mappings

#This project uses a Logistic Regression classifier, which is a widely used linear model for binary classification problems like churn prediction.

Why Logistic Regression?

Easy to interpret

Works well with linearly separable data

Efficient on large datasets

Stable, reliable probability outputs

#Key Steps in the ML Pipeline

Handling missing data; the missing values where the churn reason which was excluded from the model training

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
#Features
 Web UI

Users can input customer data through a Bootstrap-powered HTML form.
The result page shows:

Churn / Not Churn prediction

Colored text (red or green)

Probability of churn

#REST API

Send JSON to /predict and receive:

Predicted class (0 = Not Churn, 1 = Churn)

Churn probability

Flask-Based Backend

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

🔧 Installation & Setup
1️. Install dependencies
pip install flask pandas scikit-learn joblib

2️. Ensure your ML artifacts are present

churn_model.pkl

scaler.pkl

model_columns.json

3️. Run the Flask app
python app.py

4️. Open the application
http://127.0.0.1:5000/

API Usage
Endpoint
POST /predict

Example JSON Request
{
  "Tenure_Months": 18,
  "Monthly_Charges": 72.3,
  "Churn_Score": 41,
  "CLTV": 3980,
  "Churn_Value": 0
}

Example of JSON Response

(Logistic Regression output)

{
  "prediction": 1,
  "probability": 0.8124
}

#Frontend Pages
index.html

Customer input form

Bootstrap & custom CSS

Displays project image (churn.jpg)

result.html

Shows prediction

= Churn

= Not Churn

Displays churn probability

Button to go back to home

#Image Asset

Place your image inside:

static/img/churn.jpg


Displayed at the top of the homepage.


#Model Notes (Logistic Regression Insights)

Logistic Regression outputs probabilities using the sigmoid function

Ideal for churn classification due to:

Binary nature (0 or 1)

Well-calibrated probability outputs

Coefficients help interpret feature impact on churn

Works best when:

Features are scaled

Data is not highly nonlinear

Multicollinearity is handled

The model in this project is trained with all necessary preprocessing.

#Contributions

You can contribute by:

Improving the model

Adding new features

Enhancing UI

Writing tests

Pull requests are welcome.

#Acknowledgements

Flask

Scikit-learn

Bootstrap CSS

Logistic Regression theory

Pandas + joblib for ML integration

#References

Ouko, Allan. Customer Churn Prediction. github.com/AllanOuko/customer-churn-prediction-application/tree/main.

#App Input User Interface

<img width="971" height="711" alt="Screenshot 2025-11-28 190536" src="https://github.com/user-attachments/assets/3eacb821-d24a-4226-99b8-e107156d7c91" />

#App Result user Interface

<img width="1886" height="964" alt="Screenshot 2025-11-28 190606" src="https://github.com/user-attachments/assets/0369badf-179b-4812-9ae0-645ce4d0eb8c" />

