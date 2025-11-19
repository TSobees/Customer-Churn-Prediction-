# Customer-Churn-Prediction-
A machine learning project to predict customer churn integrating Web API FastAPI

Customer Churn Prediction System (ML Model + FastAPI Web API)

This project provides an end-to-end Machine Learning solution for predicting customer churn. It includes:
Data preprocessing & feature engineering
Model training & evaluation
SHAP model interpretability
A production-ready FastAPI Web API for real-time predictions
Saved model artifacts (.pkl files) and JSON feature mappings

This project uses a Logistic Regression classifier, which is a widely used linear model for binary classification problems like churn prediction.

Why Logistic Regression?

Easy to interpret

Works well with linearly separable data

Efficient on large datasets

Stable, reliable probability outputs

Key Steps in the ML Pipeline

Handling missing data; the missing values where the churn reason which was excluded from the model training

Label encoding & One-Hot Encoding

Standardizing numeric features

Preserving feature order during inference

Training a Logistic Regression classifier
