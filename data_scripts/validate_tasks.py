import asyncio
import json
import numpy as np
from tqdm import tqdm

from tools.create_data import create_seed

with open('task_vol.json') as f:
    data = json.load(f)

task_vol_new = {}

def validate_seed_(task):
    seed, _ = create_seed(task, '.')
    if np.all(seed == 0):
        return None
    else:
        return task

async def validate_seed(tasks):
    loop = asyncio.get_event_loop()
    futures = [
        loop.run_in_executor(
            None,
            validate_seed_,
            task
        )
        for task in data
    ]
    print("starting gathering")
    responses = [await f for f in tqdm(asyncio.as_completed(futures), total=len(futures))]
    for task in tqdm(responses):
        if task is not None:
            task_vol_new[task] = data[task]

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(validate_seed(data))
    with open('task_vol_new.json', 'w') as f:
        json.dump(task_vol_new, f)