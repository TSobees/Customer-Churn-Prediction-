"""Optional FastAPI adapter: python -m uvicorn main:app --host 127.0.0.1."""
from fastapi import FastAPI,Body,HTTPException
from churn.inference import predict,load_bundle
from churn.schema import EXAMPLE

app=FastAPI(title='Customer Churn Demo',description='Educational snapshot classification; no validated future horizon.')

@app.get('/health')
def health():
    try: return {'ready':True,'model':load_bundle()[1]['algorithm']}
    except (ValueError,OSError) as error: raise HTTPException(status_code=503,detail=str(error)) from error

@app.post('/predict')
def predict_churn(payload:dict=Body(...,examples=[EXAMPLE])):
    try: return predict(payload)
    except ValueError as error: raise HTTPException(status_code=422,detail=str(error)) from error
    except OSError as error: raise HTTPException(status_code=503,detail='Model unavailable. Run python -m churn.train.') from error
