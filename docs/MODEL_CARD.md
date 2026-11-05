# Churn model card

## Intended use

An educational baseline for binary classification of the provided telco snapshot.
The output estimates a class probability from customer/service attributes. There
is no validated future time horizon, causal explanation or intervention effect.

## Training and inference

`python -m churn.train` fits a fixed logistic regression pipeline. Numeric median
imputation, scaling, categorical imputation and one-hot encoding are fit on the
training subset only. A fixed stratified 80/20 split uses random state 42. The
threshold is 0.5; no test-set threshold search is performed. Flask and FastAPI use
the same pipeline and schema. `models/metadata.json` records inputs, exclusions,
versions, source fingerprint, split sizes, confusion matrix and held-out metrics.

## Changes from the earlier version

Earlier training encoded IDs and included `Churn Value`, an outcome indicator,
and churn-related score/reason fields. Those artifacts are replaced by a compact
pipeline using the explicit input allowlist in `churn/schema.py`. Old files remain
in Git history for provenance; their metrics are not a valid comparison with this
revision. Existing clients must adopt the new schema, using the JSON example in
the README or FastAPI documentation. Unsupported fields are rejected.

## Limits

This is one random holdout from one snapshot. It does not establish future, external,
fairness, calibration or business-effectiveness performance. Gender and senior-status
features are present in this teaching baseline; no fairness assessment is established.
Class probabilities are not guaranteed calibrated. Protected traits and customer
actions need separate review before any real use. Do not automatically decide
customer treatment from this demonstration. Inputs are not persisted by this app.

Only load trusted saved models. Dependency versions can affect serialized models;
the application verifies the scikit-learn version or requests retraining.

Reference: [scikit-learn data leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
