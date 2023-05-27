import requests
import os
import json

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

response1 = requests.post(url=URL + '/prediction')
print(response1)

response2 = requests.get(url=URL + "/scoring")

response3 = requests.get(url=URL + "/summarystats")
response4 = requests.get(url=URL + "/diagnostics")

responses = {'predictions': response1.text,
             'scoring': response2.text,
             'summarystats': response3.text,
             'diagnostics': response4.text}

with open('config.json', 'r') as f:
    config = json.load(f)

prodpath = os.path.join(config['prod_deployment_path'])
# write the responses to your workspace

with open(os.path.join(prodpath, 'apireturns.txt'), "w") as pf:
    pf.write(str(responses))