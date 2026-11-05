"""One inference implementation for the Flask and FastAPI applications."""
from functools import lru_cache
import json
from pathlib import Path
import joblib
import sklearn
from .schema import FEATURES,validate

ROOT=Path(__file__).resolve().parents[1]

@lru_cache(maxsize=4)
def load_bundle(directory=None):
    directory=Path(directory or ROOT/'models')
    metadata=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    if metadata['features']!=FEATURES: raise ValueError('Model input schema differs from this app. Retrain with python -m churn.train.')
    if metadata['sklearn_version']!=sklearn.__version__: raise ValueError('Install the pinned scikit-learn version or retrain with python -m churn.train.')
    # Only load the trusted repository artifact; no model-upload endpoint exists.
    model=joblib.load(directory/'churn_pipeline.joblib')
    return model,metadata

def predict(payload,directory=None):
    frame=validate(payload);model,metadata=load_bundle(directory)
    probability=float(model.predict_proba(frame)[0,1]);decision=int(probability>=metadata['threshold'])
    return {'prediction':decision,'label':'Churn' if decision else 'Not churn','probability':probability,
        'threshold':metadata['threshold'],'model':'LogisticRegression',
        'scope':'Educational classification estimate; no future churn date or validated forecast horizon.'}
