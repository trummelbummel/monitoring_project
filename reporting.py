import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

##############Function for reporting
def score_model(model_path, test_data_path):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace

    dftest = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    y_test = dftest['exited']
    X_test = dftest.drop(['exited', 'corporation'], axis=1)
    lr = pickle.load(open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb'))
    y_pred = lr.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    plt.matshow(cm)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig(os.path.join(model_path, 'confusion_matrix.png'))


if __name__ == '__main__':
    ###############Load config.json and get path variables
    with open('config.json', 'r') as f:
        config = json.load(f)

    testdata_path = os.path.join(config['test_data_path'])
    model_path = os.path.join(config['output_model_path'])
    score_model(model_path, testdata_path)
