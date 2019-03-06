from create_subset import subset_cells
from tools.clean_data import clean as clean
import os
import asyncio
import pandas as pd
import json

df = pd.read_csv('cells.csv')

df_train = df[df['Cell ID'] < 77960]
df_test = df[df['Cell ID'] >= 77960]

train_cells = df_train['Cell ID'].tolist()
test_cells = df_test['Cell ID'].tolist()

loop = asyncio.get_event_loop()

problems = loop.run_until_complete(subset_cells(train_cells, 'train_data', False))
problems2 = loop.run_until_complete(subset_cells(test_cells, 'test_data', False))

problems.extend(problems2)

with open('problems.json','w') as f:
    json.dump(problems, f)
clean(os.path.join('train_data', 'task_data.json'), os.path.join('train_data', 'task_vol.json'))
clean(os.path.join('test_data', 'task_data.json'), os.path.join('test_data', 'task_vol.json'))