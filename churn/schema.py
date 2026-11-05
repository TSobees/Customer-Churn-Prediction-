"""Inputs available before the observed churn outcome; identifiers are excluded."""
import math
import pandas as pd

NUMERIC=['Tenure Months','Monthly Charges','Total Charges']
YES_NO=['Senior Citizen','Partner','Dependents','Phone Service','Paperless Billing']
INTERNET_OPTIONS=['Online Security','Online Backup','Device Protection','Tech Support','Streaming TV','Streaming Movies']
CHOICES={
    'Gender':['Female','Male'],
    **{name:['No','Yes'] for name in YES_NO},
    'Multiple Lines':['No','No phone service','Yes'],
    'Internet Service':['No','DSL','Fiber optic'],
    **{name:['No','No internet service','Yes'] for name in INTERNET_OPTIONS},
    'Contract':['Month-to-month','One year','Two year'],
    'Payment Method':['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)']}
FEATURES=NUMERIC+list(CHOICES)
EXCLUDED=['CustomerID','Count','Country','State','City','Zip Code','Lat Long','Latitude','Longitude',
    'Churn Label','Churn Value','Churn Reason','Churn Score','CLTV']
EXAMPLE={
    'Tenure Months':18,'Monthly Charges':72.3,'Total Charges':1301.4,
    'Gender':'Female','Senior Citizen':'No','Partner':'Yes','Dependents':'No',
    'Phone Service':'Yes','Multiple Lines':'No','Internet Service':'Fiber optic',
    'Online Security':'No','Online Backup':'Yes','Device Protection':'No','Tech Support':'No',
    'Streaming TV':'No','Streaming Movies':'No','Contract':'Month-to-month',
    'Paperless Billing':'Yes','Payment Method':'Electronic check'}

def validate(payload):
    if not isinstance(payload,dict): raise ValueError('Supply a JSON object with customer inputs.')
    unknown=set(payload)-set(FEATURES)
    if unknown: raise ValueError('Unsupported fields: '+', '.join(sorted(unknown)))
    missing=[c for c in FEATURES if c not in payload and c!='Total Charges']
    if missing: raise ValueError('Missing fields: '+', '.join(missing))
    row={}
    for name in NUMERIC:
        value=payload.get(name)
        if name=='Total Charges' and (value is None or value==''):
            row[name]=float('nan');continue
        if isinstance(value,bool): raise ValueError(name+' must be a number.')
        try: number=float(value)
        except (ValueError,TypeError): raise ValueError(name+' must be a number.') from None
        if not math.isfinite(number) or number<0 or number>10_000_000:
            raise ValueError(name+' must be a finite nonnegative number no greater than 10 million.')
        if name=='Tenure Months' and number!=int(number): raise ValueError('Tenure Months must be a whole number.')
        row[name]=number
    for name,choices in CHOICES.items():
        value=payload[name]
        if value not in choices: raise ValueError(name+' must be one of: '+', '.join(choices))
        row[name]=value
    if row['Phone Service']=='No' and row['Multiple Lines']!='No phone service':
        raise ValueError('Multiple Lines must be No phone service when Phone Service is No.')
    if row['Phone Service']=='Yes' and row['Multiple Lines']=='No phone service':
        raise ValueError('Choose Yes or No for Multiple Lines when Phone Service is Yes.')
    if row['Internet Service']=='No' and any(row[n]!='No internet service' for n in INTERNET_OPTIONS):
        raise ValueError('Choose No internet service for internet add-ons when Internet Service is No.')
    if row['Internet Service']!='No' and any(row[n]=='No internet service' for n in INTERNET_OPTIONS):
        raise ValueError('Choose Yes or No for internet add-ons when internet service is active.')
    return pd.DataFrame([row],columns=FEATURES)
