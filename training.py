from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

#################Function for training the model
def train_model(model_path, dataset_csv_path):
    
    #use this logistic regression for training
    lr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100, multi_class='ovr',
                    n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'), index_col=0)
    y = df['exited']
    X = df.drop(['exited', 'corporation'], axis=1)
    lr.fit(X, y)

    #write the trained model to your workspace in a file called trainedmodel.pkl
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'wb') as f:
        pickle.dump(lr, f)


if __name__ == '__main__':
    ###################Load config.json and get path variables
    with open('config.json', 'r') as f:
        config = json.load(f)

    dataset_csv_path = os.path.join(config['output_folder_path'])
    model_path = os.path.join(config['output_model_path'])
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    train_model(model_path, dataset_csv_path)