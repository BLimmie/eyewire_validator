import os
import json
import requests
import shutil
from tqdm import tqdm
import asyncio

access_appendix = "?access_token={}".format(os.environ["EW_OAUTH2"])

cell_url_base = "https://eyewire.org/1.0/cell/{}/tasks"+access_appendix


with open('task_vol.json') as f:
    task_vols_full = json.load(f)

print('task_vol loaded')
with open('task_data.json') as f:
    print('loading task_data')
    task_data_full = json.load(f)

print("jsons loaded")
def copy_vol(vol, folder_name):
    images_path = os.path.join('images',vol)
    if os.path.isdir(os.path.join(folder_name,'images',vol)):
        shutil.rmtree(os.path.join(folder_name,'images',vol))
    shutil.copytree(images_path, os.path.join(folder_name, 'images', vol))
    seg_path = os.path.join('segmentation',vol)
    if os.path.isdir(os.path.join(folder_name,'segmentation',vol)):
        shutil.rmtree(os.path.join(folder_name,'segmentation',vol))
    shutil.copytree(seg_path, os.path.join(folder_name, 'segmentation', vol))


async def subset_cells(cell_ids, folder_name, files=True):
    problems = []
    task_vols_new = {}
    task_data_new = {}
    vols = []
    loop = asyncio.get_event_loop()
    futures = [
        loop.run_in_executor(
            None,
            requests.get,
            cell_url_base.format(int(cell_id))
        )
        for cell_id in cell_ids
    ]
    for r in tqdm(await asyncio.gather(*futures)):
        d = r.json()
        task_vols_subset = {}
        task_data_subset = {}
        for task in d["tasks"]:
            try:
                task_vols_subset[str(task['id'])] = task_vols_full[str(task['id'])]
            except:
                problems.append(task['id'])
                print(task['id'])
        
        task_vols_new.update(task_vols_subset)
        for task in d["tasks"]:
            try:
                task_data_subset[str(task['id'])] = task_data_full[str(task['id'])]
            except:
                problems.append(task['id'])
                print(task['id'])
        task_data_new.update(task_data_subset)
        
        
        vols.extend(list(set(task_vols_subset.values())))
        vols = list(set(vols))
    with open(os.path.join(folder_name,'task_vol.json'),'w') as f:
        json.dump(task_vols_new, f)
    with open(os.path.join(folder_name,'task_data.json'),'w') as f:
        json.dump(task_data_new, f)
    if files:
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(
                None,
                copy_vol,
                vol, folder_name
            )
            for vol in vols
        ]
        for r in tqdm(await asyncio.gather(*futures)):
            continue
    return list(set(problems))

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(subset_cells([1795],'dev'))