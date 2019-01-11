import lzma
import requests
import json
import numpy as np
import os

access_appendix = "?access_token={}".format(os.environ["EW_OAUTH2"])
base_url = "https://eyewire.org/1.0/task/{}"

def create_gt(task_id):
    r = requests.get(base_url.format(task_id)+access_appendix)
    data = r.json()
    vol_id = data["data"]["segmentation"]["id"]
    segmentation_path = "./segmentation/{}/segmentation.lzma".format(vol_id)
    r = requests.get(base_url.format(task_id)+"/aggregate"+access_appendix)
    data = r.json()
    segments = [int(x) for x in data["segments"].keys()]
    with lzma.open(segmentation_path) as lzma_file:
        pixels = np.frombuffer(lzma_file.read(),dtype=np.uint16).reshape((256,256,256))
    print("Read lzma")
    pixels = np.where(np.isin(pixels, segments), 1, 0)
    return pixels

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    ans = create_gt(10531)
    print(ans.shape)
    print(np.count_nonzero(ans))
    fig = plt.figure()
    ans = ans[::8,::8,::8]
    ax = fig.gca(projection='3d')

    ax.voxels(ans)
    plt.show()