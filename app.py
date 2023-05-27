from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
from diagnostics import model_predictions, execution_time, dataframe_summary, compute_na
import json
import os

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['prod_deployment_path'])
global PREDICTION_MODEL
global TESTDATA
global TESTLABELS
PREDICTION_MODEL = pickle.load(open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb'))
dftest = pd.read_csv(os.path.join(dataset_csv_path, 'testdata.csv'))
TESTLABELS = dftest['exited']
TESTDATA = dftest


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    predictions = model_predictions(TESTDATA, PREDICTION_MODEL)
    return json.dumps({'predictions': predictions})


#######################Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    predictions = model_predictions(TESTDATA, PREDICTION_MODEL)
    f1 = metrics.f1_score(TESTLABELS, predictions)
    return json.dumps({'f1': f1})


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summarystats():
    summarystats = dataframe_summary(TESTDATA)
    return json.dumps({'summarystats': summarystats.to_json()})


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    percent_na = compute_na(TESTDATA)
    timing = execution_time()
    return json.dumps({'timing': timing,
                       'percent_na': percent_na.tolist()})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
