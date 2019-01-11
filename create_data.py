import lzma
import requests
import json
import numpy as np
import os
import cv2

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
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    ans = create_gt(10531)
    print(ans.shape)
    print(np.count_nonzero(ans))
    fig = plt.figure()
    ans = ans[::4,::4,::4]
    ax = fig.gca(projection='3d')