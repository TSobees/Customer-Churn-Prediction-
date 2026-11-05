"""Train a fixed baseline using only a stratified training subset for preprocessing."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from .schema import NUMERIC,CHOICES,FEATURES,EXCLUDED

ROOT=Path(__file__).resolve().parents[1]

def train(data_path=None,output_dir=None):
    source=Path(data_path or ROOT/'data/Telco_customer_churn.csv')
    out=Path(output_dir or ROOT/'models')
    frame=pd.read_csv(source)
    missing=set(FEATURES+['Churn Label'])-set(frame)
    if missing: raise ValueError('Missing training columns: '+', '.join(sorted(missing)))
    if frame['CustomerID'].duplicated().any(): raise ValueError('Customer IDs must be unique before splitting.')
    y=frame['Churn Label'].map({'Yes':1,'No':0})
    if y.isna().any(): raise ValueError('Churn Label must be Yes or No.')
    x=frame[FEATURES].copy()
    for name in NUMERIC: x[name]=pd.to_numeric(x[name],errors='coerce')
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.2,random_state=42,stratify=y)
    numeric=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    categorical=Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))])
    preprocessing=ColumnTransformer([('numeric',numeric,NUMERIC),('categorical',categorical,list(CHOICES))])
    model=Pipeline([('prepare',preprocessing),('classifier',LogisticRegression(max_iter=2000,random_state=42))])
    model.fit(xtrain,ytrain)
    probability=model.predict_proba(xtest)[:,1];prediction=(probability>=.5).astype(int)
    metrics={name:float(fn(ytest,prediction,zero_division=0)) for name,fn in [('precision',precision_score),('recall',recall_score),('f1',f1_score)]}
    metrics.update(accuracy=float(accuracy_score(ytest,prediction)),roc_auc=float(roc_auc_score(ytest,probability)),
        average_precision=float(average_precision_score(ytest,probability)),test_prevalence=float(ytest.mean()),
        confusion_matrix=confusion_matrix(ytest,prediction,labels=[0,1]).tolist())
    metadata={'schema_version':1,'algorithm':'LogisticRegression','threshold':.5,
        'features':FEATURES,'excluded_columns':EXCLUDED,'rows':len(frame),'train_rows':len(xtrain),'test_rows':len(xtest),
        'split':'Stratified random 80/20 split, random_state=42; no temporal or external validation.',
        'dataset_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'sklearn_version':sklearn.__version__,
        'python_version':platform.python_version(),'metrics':metrics,
        'limitations':'Educational snapshot classifier. Labels have no event time; risk is not tied to a future horizon. No field, temporal, fairness or probability-calibration validation is claimed.'}
    out.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,out/'churn_pipeline.joblib')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path);parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args(argv);print(json.dumps(train(args.data,args.output_dir),indent=2))

if __name__=='__main__': main()
