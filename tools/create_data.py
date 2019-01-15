import lzma
import requests
import json
import numpy as np
import os
import cv2

from cube_permutations import modulations

access_appendix = "?access_token={}".format(os.environ["EW_OAUTH2"])
base_url = "https://eyewire.org/1.0/task/{}"

def create_gt(task_id):
    r = requests.get(base_url.format(task_id)+access_appendix)
    data = r.json()
    vol_id = data["data"]["segmentation"]["id"]
    segmentation_path = os.path.join("segmentation",str(vol_id),"segmentation.lzma")
    r = requests.get(base_url.format(task_id)+"/aggregate"+access_appendix)
    data = r.json()
    segments = [int(x) for x in data["segments"].keys()]
    with lzma.open(segmentation_path) as lzma_file:
        pixels = np.frombuffer(lzma_file.read(),dtype=np.uint16).reshape((256,256,256))
    print("Read lzma")
    pixels = np.where(np.isin(pixels, segments), 1, 0)
    return pixels

def create_seed(task_id):
    r = requests.get(base_url.format(task_id)+access_appendix)
    data = r.json()
    vol_id = data["data"]["segmentation"]["id"]
    segmentation_path = os.path.join("segmentation",str(vol_id),"segmentation.lzma")

    segments = [int(x) for x in data["prior"]["segments"].keys()]
    with lzma.open(segmentation_path) as lzma_file:
        pixels = np.frombuffer(lzma_file.read(),dtype=np.uint16).reshape((256,256,256))
    print("Read lzma")
    pixels = np.where(np.isin(pixels, segments), 1, 0)
    return pixels

def create_3d_image(task_id):
    r = requests.get(base_url.format(task_id)+access_appendix)
    data = r.json()
    vol_id = data["data"]["segmentation"]["id"]
    images_path = os.path.join("images", str(vol_id))
    full_image = np.zeros((256,256,256))
    for i in range(256):
        i_image_path = os.path.join(images_path, "{}.jpg".format(i))
        image_array = cv2.imread(i_image_path,0)
        full_image[i,:,:] = image_array
    return full_image
        
def create_perm_stack(task_id):
    img = create_3d_image(task_id)
    seed = create_seed(task_id)
    gt = create_gt(task_id)
    img_perm = modulations(img)
    seed_perm = modulations(seed)
    gt_perm = modulations(gt)
    for i, s, g in zip(img_perm, seed_perm, gt_perm):
        m = np.zeros((2,256,256,256))
        m[0,:,:,:] = i
        m[1,:,:,:] = s
        yield m, g

def create_discrim_stack(task_id):
    img = create_3d_image(task_id)
    seed = create_seed(task_id)
    gt = create_gt(task_id)
    m = np.zeros((3,256,256,256))
    m[0,:,:,:] = img
    m[1,:,:,:] = seed
    m[2,:,:,;] = gt
    return m
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    ans = create_gt(10531)
    seed = create_seed(10531)
    print(ans.shape)
    print(np.count_nonzero(ans))
    print(seed.shape)
    print(np.count_nonzero(seed))
    # fig = plt.figure()
    # ans = ans[::4,::4,::4]
    # ax = fig.gca(projection='3d')
    img = create_3d_image(10531)
    base_gt_slice = seed[1]
    base_slice = img[1]
    print(ans[1])
    mask = (base_gt_slice == 1)
    base_slice_copy = np.copy(base_slice)
    base_slice_copy[mask] = 255
    implot = plt.imshow(base_slice_copy, cmap='gray')
    plt.show()
