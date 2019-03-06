import pandas as pd
import asyncio

import get_data

completed_cells = []

def get_completed_cells(filename):
    with open(filename) as f:
        for line in f.readlines():
            completed_cells.append(int(line))    

df = pd.read_csv("cells.csv")
get_completed_cells('done_cells.txt')

for cell_id in df["Cell ID"]:
    if cell_id in completed_cells:
        continue
    loop = asyncio.get_event_loop()
    loop.run_until_complete(get_data.download_cell(int(cell_id)))
    with open('done_cells.txt', 'a') as f:
        f.write("{}\n".format(int(cell_id)))