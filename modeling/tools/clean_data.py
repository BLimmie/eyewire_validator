import json

def clean(task_data_filename, task_vol_filename):
    to_remove = []
    with open(task_data_filename) as f:
        task_data = json.load(f)
    with open(task_vol_filename) as f:
        task_vol = json.load(f)

    for task in task_data:
        if len(task_data[task]["aggregate"]) == 0:
            to_remove.append(task)
    for task in to_remove:
        try:
            task_data.pop(task)
        except:
            pass
        try:
            task_vol.pop(task)
        except:
            pass
    with open(task_data_filename, 'w') as f:
        json.dump(task_data, f)
    with open(task_vol_filename, 'w') as f:
        json.dump(task_vol, f)