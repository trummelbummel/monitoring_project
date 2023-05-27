
import json
import os
import subprocess
import training
import scoring
import deployment
import diagnostics
import reporting
import logging


def ingest_new_files(input_folder_path, newdata_path):
    with open(os.path.join(input_folder_path, 'ingestedfiles.txt'), 'r') as f:
        files = f.readlines()
    for file in os.listdir(newdata_path):
        if file not in files:
            subprocess.call(['python', 'ingestion.py'])
            return True
        else:
            logging.info('SUCCESS: Completed process no new files found.')
            return False

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
def check_model_drift(model_path, test_data_path, prod_path):
    with open(os.path.join(prod_path, 'latestscore.txt'), 'r') as f:
        oldf1 = f.read()

    newf1 = scoring.score_model(test_data_path, model_path)
    if float(oldf1) > newf1:
        logging.info('Model drift has occured.')
        return True
    else:
        return False

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here



##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model



if __name__ == '__main__':
    ##################Load config.json and correct path variable
    with open('config.json', 'r') as f:
        config = json.load(f)

    new_data_path = os.path.join(config['input_folder_path'])
    input_path  = os.path.join(config['output_folder_path'])
    model_path = os.path.join(config['output_model_path'])
    prod_deployment_path = os.path.join(config['prod_deployment_path'])
    test_data_path = os.path.join(config['test_data_path'])
    ingested_bool = ingest_new_files(input_path, new_data_path)

    if ingested_bool:

        training_bool = check_model_drift(model_path, test_data_path, prod_deployment_path)
        if training_bool:
            subprocess.call(['python', 'training.py'])
            subprocess.call(['python', 'deployment.py'])
            subprocess.call(['python', 'reporting.py'])
            subprocess.call(['python', 'apicalls.py'])
        else:
            logging.info('SUCCESS: Terminating process. No model deployment necessary.')







