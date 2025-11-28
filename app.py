# --------------------------
# IMPORTS
# --------------------------
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import json

# --------------------------
# INITIALIZE FLASK APP
# --------------------------
app = Flask(__name__)

# --------------------------
# LOAD MODEL, SCALER, AND COLUMNS
# --------------------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

with open("model_columns.json", "r") as f:
    model_columns = json.load(f)


# ---------------------------------------------------
# STEP 2: HOME ROUTE  (Fixes 404 error)
# ---------------------------------------------------
@app.route("/", methods=["GET"])
def home_page():
    return render_template("index.html")


# ---------------------------------------------------
# HTML FORM PREDICTION ROUTE
# ---------------------------------------------------
@app.route("/form_predict", methods=["POST"])
def form_predict():

    # Collect form inputs
    data = {
        "Tenure_Months": float(request.form.get("Tenure_Months")),
        "Monthly_Charges": float(request.form.get("Monthly_Charges")),
        "Churn_Score": float(request.form.get("Churn_Score")),
        "CLTV": float(request.form.get("CLTV")),
        "Churn_Value": float(request.form.get("Churn_Value"))
    }

    # Convert to DataFrame
    input_cp = pd.DataFrame([data])

    # Rename fields to training column names
    rename_map = {
        "Tenure_Months": "Tenure Months",
        "Monthly_Charges": "Monthly Charges",
        "Churn_Score": "Churn Score",
        "CLTV": "CLTV",
        "Churn_Value": "Churn Value"
    }
    input_cp = input_cp.rename(columns=rename_map)

    # Add missing one-hot columns
    for col in model_columns:
        if col not in input_cp:
            input_cp[col] = 0

    # Reorder
    input_cp = input_cp[model_columns]

    # Scale numerics
    num_cols = ["Tenure Months", "Monthly Charges", "Churn Score", "CLTV", "Churn Value"]
    input_cp[num_cols] = scaler.transform(input_cp[num_cols])

    # Predict
    prediction = int(model.predict(input_cp)[0])
    probability = float(model.predict_proba(input_cp)[0][1])

    result_text = "Churn" if prediction == 1 else "Not Churn"

    return render_template(
        "result.html",
        data={
            "prediction": result_text,
            "predict_probabality": round(probability, 4)
        }
    )


# ---------------------------------------------------
# ORIGINAL JSON API PREDICTION ROUTE (still works)
# ---------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict_churn():

    data = request.get_json()
    input_cp = pd.DataFrame([data])

    rename_map = {
        "Tenure_Months": "Tenure Months",
        "Monthly_Charges": "Monthly Charges",
        "Churn_Score": "Churn Score",
        "CLTV": "CLTV",
        "Churn_Value": "Churn Value"
    }

    input_cp = input_cp.rename(columns=rename_map)

    for col in model_columns:
        if col not in input_cp:
            input_cp[col] = 0

    input_cp = input_cp[model_columns]

    num_cols = ["Tenure Months", "Monthly Charges", "Churn Score", "CLTV", "Churn Value"]
    input_cp[num_cols] = scaler.transform(input_cp[num_cols])

    prediction = int(model.predict(input_cp)[0])
    probability = float(model.predict_proba(input_cp)[0][1])

    return jsonify({
        "prediction": prediction,
        "probability": probability
    })


# --------------------------
# RUN FLASK APP
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)
