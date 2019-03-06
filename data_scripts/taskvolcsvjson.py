import csv
import json

j = {}
with open('task_vol.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        j[str(row[0])] = row[1]

with open('task_vol.json','w') as f:
    json.dump(j, f)