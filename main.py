from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json

app = FastAPI(title="Customer Churn Prediction API")

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load final model column order
with open("model_columns.json", "r") as f:
    model_columns = json.load(f)

# Pydantic input model (API receives these)
class Customer(BaseModel):
    Tenure_Months: float
    Monthly_Charges: float
    Churn_Score: float
    CLTV: float
    Churn_Value: float


@app.post("/predict")
def predict_churn(customer: Customer):

    # Convert API input → Pandas DataFrame
    input_cp = pd.DataFrame([customer.dict()])

    # Rename API fields to match TRAINING columns
    rename_map = {
        "Tenure_Months": "Tenure Months",
        "Monthly_Charges": "Monthly Charges",
        "Churn_Score": "Churn Score",
        "CLTV": "CLTV",
        "Churn_Value": "Churn Value"
    }

    input_cp = input_cp.rename(columns=rename_map)

    # Add missing one-hot encoded columns
    for col in model_columns:
        if col not in input_cp:
            input_cp[col] = 0

    # Reorder columns to match model training
    input_cp = input_cp[model_columns]

    # Identify numeric columns for scaling
    numeric_cols = ["Tenure Months", "Monthly Charges", "Churn Score", "CLTV", "Churn Value"]

    # Scale numerical features
    input_cp[numeric_cols] = scaler.transform(input_cp[numeric_cols])

    # Predict
    prediction = int(model.predict(input_cp)[0])
    probability = float(model.predict_proba(input_cp)[0][1])

    return {
        "prediction": prediction,
        "probability": probability
    }


#To successfully generate link fron terminal use 
#python -m uvicorn main:app --reload
