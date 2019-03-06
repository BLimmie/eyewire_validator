import json
import pandas as pd
import multiprocessing as mp
import requests
from time import sleep
import os
from tqdm import tqdm
data = {}
with open("task_data.json") as f:
    data = json.load(f)

access_appendix = "?access_token={}".format(os.environ["EW_OAUTH2"])
task_url_base = "https://eyewire.org/1.0/task/{}"+access_appendix

def retrieve(task_id):
    url = task_url_base.format(task_id)
    agg_url = task_url_base.format(str(task_id)+'/aggregate')
    r = requests.get(url)
    task_data = r.json()
    r1 = requests.get(agg_url)
    task_agg_data = r1.json()
    j = {}
    j["seed"] = list(task_data["prior"]["segments"].keys())
    j["parent"] = task_data["parent"]
    j["aggregate"] = list(task_agg_data["segments"].keys())
    sleep(.1)
    return (str(task_id), j)

tasks = data.keys()
df = pd.read_csv("task_vol.csv", header=None)
print(df.tail())

df[0] = df[0].astype(str)
df1 = df[~df[0].isin(tasks)]
have_yet_tasks = df1[0].tolist()
print(len(have_yet_tasks))
for i in tqdm(range(int(len(have_yet_tasks)/5000)+1)):
    if (i+1)*5000 >= len(have_yet_tasks):
        current_task = have_yet_tasks[i*5000:]
    else:
        current_task = have_yet_tasks[i*5000:(i+1)*5000]
    with mp.Pool(8) as p:
        results = list(tqdm(p.imap(retrieve,current_task)))
    for (task_id, j) in results:
        data[task_id] = j
    with open("task_data.json", 'w') as f:
        json.dump(data, f)


