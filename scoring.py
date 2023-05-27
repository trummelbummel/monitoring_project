from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json




#################Function for model scoring
def score_model(test_data_path, model_path):
    dftest = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    y_test = dftest['exited']
    X_test = dftest.drop(['exited', 'corporation'], axis=1)
    lr = pickle.load(open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb'))

    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    y_pred = lr.predict(X_test)
    f1 = metrics.f1_score(y_pred, y_test)
    with open(os.path.join(model_path , 'latestscore.txt'), 'w') as f:
        f.writelines(str(f1))
    return f1


if __name__ == '__main__':
    #################Load config.json and get path variables
    with open('config.json','r') as f:
        config = json.load(f)

    dataset_csv_path = os.path.join(config['output_folder_path'])
    test_data_path = os.path.join(config['test_data_path'])
    model_path = os.path.join(config['output_model_path'])
    score_model(test_data_path, model_path)

