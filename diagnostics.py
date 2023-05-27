
import pandas as pd
import numpy as np
import timeit
import subprocess
import pickle
import os
import json
import copy


##################Function to get model predictions
def model_predictions(testdata, model):
    #read the deployed model and a test dataset, calculate predictions
    X = copy.deepcopy(testdata).drop(['exited', 'corporation'], axis=1)
    y_pred = model.predict(X)
    return y_pred.tolist()

##################Function to get summary statistics
def dataframe_summary(df):
    dfnumeric = copy.deepcopy(df).drop(['exited', 'corporation'], axis=1)
    #calculate summary statistics here
    return dfnumeric.describe()

##################Function to get timings
def execution_time():
    timeinglist = list()
    #calculate timing of training.py and ingestion.py
    starttime = timeit.timeit()
    subprocess.run(['python', 'ingestion.py'])
    endtime = timeit.timeit()
    timeinglist.append(endtime - starttime)
    starttime = timeit.timeit()
    subprocess.run(['python', 'training.py'])
    endtime = timeit.timeit()
    timeinglist.append(endtime - starttime)
    return timeinglist


def compute_na(df):
    total = len(df)
    percent_na = df.isna().sum()/total
    return percent_na

##################Function to check dependencies
def outdated_packages_list():
    #get a list of
    outdatedpacks = subprocess.check_output(['python', '-m', 'pip', '-o'])
    return outdatedpacks.split(' ')



if __name__ == '__main__':
    ##################Load config.json and get environment variables
    with open('config.json', 'r') as f:
        config = json.load(f)

    dataset_csv_path = os.path.join(config['output_folder_path'])
    prod_deployment_path = os.path.join(config['prod_deployment_path'])
    test_data_path = os.path.join(config['test_data_path'])
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    compute_na(df)
    testdata = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    lr = pickle.load(open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb'))
    model_predictions(lr, testdata, prod_deployment_path)
    dataframe_summary(testdata)
    execution_time()
    outdated_packages_list()





    
