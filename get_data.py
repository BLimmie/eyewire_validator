import requests
import json
import urllib
import os
import urllib
import urllib.request
import multiprocessing
from functools import partial
from time import sleep
from tqdm import tqdm

completed_vol = {}

def slurp(filename):
    with open(filename) as f:
        for line in f.readlines():
            task = int(line[:line.find(':')])
            vol = int(line[line.find(':')+1:])
            completed_vol[task] = vol


def retrieve(vol_id, url, image=True):
    if image:
        path = os.path.join("images", str(vol_id), url[url.rfind('/')+1:])
    else:
        path = os.path.join("segmentation", str(vol_id), url[url.rfind('/')+1:])
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
        sleep(.05)



access_appendix = "?access_token={}".format(os.environ["EW_OAUTH2"])

cell_url_base = "https://eyewire.org/1.0/cell/{}/tasks"+access_appendix
task_url_base = "https://eyewire.org/1.0/task/{}"+access_appendix

# Make base folders if necessary
if not os.path.isdir("images"):
    os.mkdir("images")
if not os.path.isdir("segmentation"):
    os.mkdir("segmentation")

# Load save
slurp('task_to_vol.txt')

def download_cell(cell_id):
    cell_url = cell_url_base.format(cell_id)
    r = requests.get(cell_url)
    data = r.json()

    for task in tqdm(data["tasks"], desc = "Volumes downloaded", leave=False):
        if task["id"] in completed_vol:
            print("Already Downloaded task_id: {}".format(task["id"]))
            continue
        task_url = task_url_base.format(str(task["id"]))
        r = requests.get(task_url)
        task_data = r.json()
        vol_id = task_data["data"]["segmentation"]["id"]

        if vol_id in completed_vol.values():
            print("Already Downloaded vol_id: {}".format(vol_id))
            with open('task_to_vol.txt','a') as f:
                f.write("{}:{}\n".format(task["id"],vol_id))
            continue

        data_path = task_data["full_path"]
        images = [data_path+"jpg/{}.jpg".format(i) for i in range(256)]
        segmentaion = data_path+"segmentation.lzma"
        if not os.path.isdir(os.path.join("segmentation",str(vol_id))):
            os.mkdir(os.path.join("segmentation",str(vol_id)))
        if not os.path.isdir(os.path.join("images",str(vol_id))):
            os.mkdir(os.path.join("images",str(vol_id)))
        # Download files
        print("Downloading vol_id: {}".format(vol_id))
        func = partial(retrieve, vol_id)
        with multiprocessing.Pool(16) as p:
            p.map(func, images)
        retrieve(vol_id, segmentaion, False)
        completed_vol[task["id"]] = vol_id
        with open('task_to_vol.txt','a') as f:
            f.write("{}:{}\n".format(task["id"],vol_id))

def download_task(task_id):
    task_url = task_url_base.format(task_id)
    r = requests.get(task_url)
    task_data = r.json()
    vol_id = task_data["data"]["segmentation"]["id"]
    data_path = task_data["full_path"]
    images = [data_path+"jpg/{}.jpg".format(i) for i in range(256)]
    segmentaion = data_path+"segmentation.lzma"
    if not os.path.isdir(os.path.join("segmentation",str(vol_id))):
        os.mkdir(os.path.join("segmentation",str(vol_id)))
    if not os.path.isdir(os.path.join("images",str(vol_id))):
        os.mkdir(os.path.join("images",str(vol_id)))
        print("Downloading vol_id: {}".format(vol_id))

    # Download files
    func = partial(retrieve, vol_id)
    with multiprocessing.Pool(16) as p:
        p.map(func, images)
    retrieve(vol_id, segmentaion, False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--cell_id", default=None)
    parser.add_argument("--task_id", default=None)
    args = parser.parse_args()

    # Assert that only one is selected
    assert (args.cell_id is not None or args.task_id is not None)
    assert (args.cell_id is None or args.task_id is None)
    
    if args.cell_id is not None:
        download_cell(args.cell_id)
    
    if args.task_id is not None:
        download_task(args.task_id)