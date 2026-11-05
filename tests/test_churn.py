import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from app import create_app
from churn.schema import EXAMPLE,FEATURES,EXCLUDED,validate
from churn.inference import predict,load_bundle

class ChurnTests(unittest.TestCase):
    def setUp(self): self.client=create_app().test_client()
    def test_model_schema_has_no_outcome_or_identifier_inputs(self):
        model,meta=load_bundle()
        self.assertFalse(set(FEATURES)&set(EXCLUDED))
        self.assertEqual(list(model.feature_names_in_),FEATURES)
        self.assertLess(len(model['prepare'].get_feature_names_out()),100)
        self.assertEqual(meta['train_rows']+meta['test_rows'],meta['rows'])
    def test_home_health_and_json(self):
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertEqual(self.client.get('/health').status_code,200)
        result=self.client.post('/predict',json=EXAMPLE)
        self.assertEqual(result.status_code,200,result.json)
        self.assertTrue(0<=result.json['probability']<=1)
        self.assertEqual(result.json['prediction'],int(result.json['probability']>=.5))
    def test_form_and_json_use_the_same_model(self):
        result=self.client.post('/form_predict',data=EXAMPLE)
        self.assertEqual(result.status_code,200)
        self.assertIn(f"{predict(EXAMPLE)['probability']*100:.1f}%",result.text)
    def test_invalid_inputs_have_clear_client_errors(self):
        for change in [{'Churn Value':1},{'Monthly Charges':-1},{'Monthly Charges':'NaN'}, {'Tenure Months':True}, {'Contract':'unknown'},{'Phone Service':'No'}]:
            with self.subTest(change=change):
                self.assertEqual(self.client.post('/predict',json={**EXAMPLE,**change}).status_code,400)
        self.assertEqual(self.client.post('/predict',json={}).status_code,400)
        self.assertEqual(self.client.post('/predict',data='bad',content_type='application/json').status_code,400)
    def test_missing_total_charges_can_be_imputed(self):
        row=EXAMPLE.copy();del row['Total Charges']
        self.assertTrue(np.isnan(validate(row)['Total Charges'][0]))
        self.assertTrue(0<=predict(row)['probability']<=1)
    def test_fastapi_matches_flask(self):
        from fastapi.testclient import TestClient
        from main import app
        with TestClient(app) as client:
            response=client.post('/predict',json=EXAMPLE)
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(response.json(),predict(EXAMPLE))
            self.assertEqual(client.post('/predict',json={'Churn Value':1}).status_code,422)
    def test_model_is_independent_of_current_working_directory(self):
        self.assertTrue((Path(__file__).resolve().parents[1]/'models/churn_pipeline.joblib').exists())
        self.assertEqual(self.client.get('/api/model-info').status_code,200)

if __name__=='__main__': unittest.main()
