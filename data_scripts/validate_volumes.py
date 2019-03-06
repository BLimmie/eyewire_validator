import json
import os
from tqdm import tqdm

from get_data import download_task

with open('task_vol.json') as f:
    data = json.load(f)

for task in tqdm(data):
    if not os.path.isdir(os.path.join('images', str(data[task]))):
        download_task(int(task))