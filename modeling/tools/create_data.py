import lzma
import requests
import json
import numpy as np
import os
import cv2

if __name__ == "__main__":
    from cube_permutations import single_perm
else:
    from tools.cube_permutations import single_perm

DEFAULT_CONFIDENCE = 10

if os.environ["EW_OAUTH2"] is not None:
    access_appendix = "?access_token={}".format(os.environ["EW_OAUTH2"])
    base_url = "https://eyewire.org/1.0/task/{}"

with open('task_vol.json') as f:
    task_vol = json.load(f)

with open('task_data.json') as f:
    full_task_data = json.load(f)

def get_parent(task_id):
    if task_id is None:
        return None
    if task_id not in full_task_data:
        return None
    return full_task_data[str(task_id)]["parent"]

def create_gt_raw(task_id, data_path):
    r = requests.get(base_url.format(task_id)+access_appendix)
    data = r.json()
    vol_id = data["data"]["segmentation"]["id"]
    segmentation_path = os.path.join(data_path, "segmentation",str(vol_id),"segmentation.lzma")
    r = requests.get(base_url.format(task_id)+"/aggregate"+access_appendix)
    data = r.json()
    segments = [int(x) for x in data["segments"].keys()]
    with lzma.open(segmentation_path) as lzma_file:
        pixels = np.frombuffer(lzma_file.read(),dtype=np.uint16).reshape((64,64,64))
    print("Read lzma")
    pixels = np.where(np.isin(pixels, segments), 1, 0)
    return pixels

def create_gt(task_id, data_path):
    if task_id is None or task_id not in task_vol or task_id not in full_task_data:
        return np.zeros((64,64,64))
    vol_id = task_vol[str(task_id)]
    segmentation_path = os.path.join(data_path, "segmentation",str(vol_id),"segmentation.lzma")
    segments = full_task_data[str(task_id)]["aggregate"]
    with lzma.open(segmentation_path) as lzma_file:
        pixels = np.frombuffer(lzma_file.read(),dtype=np.uint16).reshape((64,64,64))
    pixels = np.where(np.isin(pixels, segments), 1, 0)
    return pixels

def create_seed_raw(task_id, data_path):
    r = requests.get(base_url.format(task_id)+access_appendix)
    data = r.json()
    vol_id = data["data"]["segmentation"]["id"]
    segmentation_path = os.path.join(data_path, "segmentation",str(vol_id),"segmentation.lzma")
    segments = [int(x) for x in data["prior"]["segments"].keys()]
    with lzma.open(segmentation_path) as lzma_file:
        pixels = np.frombuffer(lzma_file.read(),dtype=np.uint16).reshape((64,64,64))
    pixels = np.where(np.isin(pixels, segments), 1, 0)
    return pixels

def create_seed(task_id, data_path):
    vol_id = task_vol[str(task_id)]
    segmentation_path = os.path.join(data_path, "segmentation",str(vol_id),"segmentation.lzma")
    segments = full_task_data[str(task_id)]["seed"]
    with lzma.open(segmentation_path) as lzma_file:
        pixels = np.frombuffer(lzma_file.read(),dtype=np.uint16).reshape((64,64,64))
    pixels = np.where(np.isin(pixels, segments), 1, 0)
    confidence = np.where(pixels==1, 1, DEFAULT_CONFIDENCE)
    return pixels, confidence

def create_3d_image_raw(task_id, data_path):
    r = requests.get(base_url.format(task_id)+access_appendix)
    data = r.json()
    vol_id = data["data"]["segmentation"]["id"]
    images_path = os.path.join(data_path, "images", str(vol_id), "image.lzma")
    with lzma.open(images_path) as lzma_file:
        full_image = np.frombuffer(lzma_file.read(),dtype=np.uint8).reshape((64,64,64))
    return full_image

def create_3d_image(task_id, data_path):
    vol_id = task_vol[str(task_id)]
    images_path = os.path.join(data_path, "images", str(vol_id), "image.lzma")
    with lzma.open(images_path) as lzma_file:
        full_image = np.frombuffer(lzma_file.read(),dtype=np.uint8).reshape((64,64,64))
    return full_image
        
def create_perm_stack(task_id, data_path):
    img = create_3d_image(task_id, data_path)
    seed, sigma = create_seed(task_id, data_path)
    gt = create_gt(task_id, data_path)
    img_perm = permutations(img)
    seed_perm = permutations(seed)
    sigma_perm = permutations(sigma)
    gt_perm = permutations(gt)
    for i, s, g in zip(img_perm, seed_perm, gt_perm):
        yield i, s, g

def create_indiv_perm_stack(task_id, data_path, idx, prev=0):
    idx = idx%96
    img = create_3d_image(task_id, data_path)
    seed, sigma = create_seed(task_id, data_path)
    gt = create_gt(task_id, data_path)
    img_perm = single_perm(img, idx)
    seed_perm = single_perm(seed, idx)
    sigma_perm = single_perm(sigma, idx)
    gt_perm = single_perm(gt, idx)
    return img_perm, seed_perm, sigma_perm, gt_perm

def create_stack(task_id, data_path, prev=0):
    img = create_3d_image(task_id, data_path)
    seed, sigma = create_seed(task_id, data_path)
    gt = create_gt(task_id, data_path)
    if prev > 0:
        r_task_id = task_id
        parents = []
        for _ in range(prev):
            parent = get_parent(r_task_id)
            parents.append(create_gt(r_task_id, data_path))
            r_task_id = parent
        parents.reverse()
        prev_gt = np.stack(parents, 0)
        prev_gt = prev_gt.astype(np.int64)
    else:
        prev_gt = []
    return img, seed, sigma, prev_gt, gt
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    ans = create_gt(10531, '')
    seed, sigma = create_seed(10531, '')
    print(ans.shape)
    print(np.count_nonzero(ans))
    print(seed.shape)
    print(np.count_nonzero(seed))
    # fig = plt.figure()
    # ans = ans[::4,::4,::4]
    # ax = fig.gca(projection='3d')
    img = create_3d_image(10531, '')
    base_gt_slice = seed[1]
    base_slice = img[1]
    cv2.imwrite('asdf.jpg', img[0])
    print(ans[1])
    mask = (base_gt_slice == 1)
    base_slice_copy = np.copy(base_slice)
    base_slice_copy[mask] = 255
    implot = plt.imshow(base_slice_copy, cmap='gray')
    plt.show()
