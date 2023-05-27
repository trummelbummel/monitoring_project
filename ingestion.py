import pandas as pd
import os
import json

#############Function for data ingestion
def merge_multiple_dataframe(input_folder_path):
    #check for datasets, compile them together, and write to an output file
    filescontents = list()
    filesread = list()
    for filename in os.listdir(input_folder_path):
        if filename.endswith('.csv'):
            filesread.append(filename)
            filescontents.append(pd.read_csv(os.path.join(input_folder_path, filename)))
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w') as f:
        f.writelines(str(filesread))
    return pd.concat(filescontents)

def deduplicate(df):
    return df.drop_duplicates()

if __name__ == '__main__':
    #############Load config.json and get input and output paths
    with open('config.json', 'r') as f:
        config = json.load(f)

    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)

    df = merge_multiple_dataframe(input_folder_path)
    df = deduplicate(df)
    df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'))


