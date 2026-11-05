"""Flask UI and JSON endpoint sharing the same validated model contract."""
from flask import Flask,request,jsonify,render_template
from churn.schema import NUMERIC,CHOICES,EXAMPLE
from churn.inference import predict,load_bundle

def create_app(model_dir=None):
    app=Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=64*1024)
    def home(error=None,values=None,status=200):
        return render_template('index.html',numeric=NUMERIC,choices=CHOICES,values=values or EXAMPLE,error=error),status
    @app.get('/')
    def index(): return home()
    @app.get('/health')
    def health():
        try:
            _,meta=load_bundle(model_dir)
            return jsonify(ready=True,model=meta['algorithm'],dataset_sha256=meta['dataset_sha256'])
        except (OSError,ValueError) as error: return jsonify(ready=False,error=str(error)),503
    @app.get('/api/model-info')
    def info():
        try: return jsonify(load_bundle(model_dir)[1])
        except (OSError,ValueError) as error: return jsonify(error=str(error)),503
    @app.post('/predict')
    def predict_json():
        try: return jsonify(predict(request.get_json(silent=True),model_dir))
        except ValueError as error: return jsonify(error=str(error)),400
        except OSError: return jsonify(error='Model unavailable. Run python -m churn.train.'),503
    @app.post('/form_predict')
    def form_predict():
        try: return render_template('result.html',result=predict(request.form.to_dict(),model_dir))
        except ValueError as error: return home(str(error),request.form.to_dict(),400)
        except OSError: return home('Model unavailable. Run python -m churn.train.',request.form.to_dict(),503)
    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options']='nosniff';response.headers['X-Frame-Options']='DENY'
        response.headers['Cache-Control']='no-store'
        response.headers['Content-Security-Policy']="default-src 'self'; style-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        return response
    return app

app=create_app()
if __name__=='__main__': app.run(host='127.0.0.1',port=5000,debug=False)
