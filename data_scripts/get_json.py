import json
import asyncio
import csv
import requests
import os
from tqdm import tqdm


access_appendix = "?access_token={}".format(os.environ["EW_OAUTH2"])
task_url_base = "https://eyewire.org/1.0/task/{}"+access_appendix
try:
    with open('task_data.json') as f:
        data = json.load(f)
except:
    data = {}

	
def get(url, task):
	r = requests.get(url)
	return (r, task)

async def run(tasks):
    loop = asyncio.get_event_loop()
    futures = [
        loop.run_in_executor(
            None,
            get,
            task_url_base.format(int(task)), int(task)
        )
        for task in tasks
    ]
    for (r, task) in await asyncio.gather(*futures):
        task_data = r.json()
        task_id = task
        data[str(task_id)] = {}
        data[str(task_id)]["seed"] = [int(x) for x in task_data["prior"]["segments"].keys()]
        data[str(task_id)]["parent"] = task_data["parent"]

    loop = asyncio.get_event_loop()
    futures = [
        loop.run_in_executor(
            None,
            get,
            task_url_base.format(str(task)+'/aggregate'), int(task)
        )
        for task in tasks
    ]
    for (r, task) in await asyncio.gather(*futures):
        task_data = r.json()
        task_id = task
        data[str(task_id)]["aggregate"] = [int(x) for x in task_data["segments"].keys()]


if __name__ == "__main__":
    tasks = []
    with open("task_vol.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            tasks.append(row[0])
    print(len(tasks))
    for i in tqdm(range(int(len(data.keys())/10000),195)):
        loop = asyncio.get_event_loop()
        if i == 194:
            loop.run_until_complete(run(tasks[i*10000:]))
        else:
            loop.run_until_complete(run(tasks[i*10000:(i+1)*10000]))
        with open("task_data.json", 'w') as f:
            json.dump(data, f)