import json
import sys


with open(sys.argv[1]) as f:
    data = json.load(f)

with open(sys.argv[2]) as f:
    data.extend(json.load(f))

data = list(set(data))

with open(sys.argv[1], 'w') as f:
    json.dump(data, f)