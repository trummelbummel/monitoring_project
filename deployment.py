from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil



####################function for deployment
def store_model_into_pickle(output_path, output_model, prodpath):
    if not os.path.exists(prodpath):
        os.mkdir(prodpath)
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    for file in os.listdir(output_model):
        shutil.copyfile(os.path.join(output_model, file), os.path.join(prodpath, file))
    # 2nd option
    for file in os.listdir(output_path):
        if '.txt' in file:
            shutil.copyfile(os.path.join(output_path, file), os.path.join(prodpath, file))


if __name__ == '__main__':
    ##################Load config.json and correct path variable
    with open('config.json', 'r') as f:
        config = json.load(f)

    output_path = os.path.join(config['output_folder_path'])
    output_model = os.path.join(config['output_model_path'])
    prod_deployment_path = os.path.join(config['prod_deployment_path'])
    store_model_into_pickle(output_path, output_model, prod_deployment_path)


        
        
        

