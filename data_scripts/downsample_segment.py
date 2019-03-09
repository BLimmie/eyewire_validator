import numpy as np
import os
import lzma
from tqdm import tqdm
import json

CKPT = 100
vols = [os.path.join('segmentation', o) for o in os.listdir(
        'segmentation') if os.path.isdir(os.path.join('segmentation', o))]
try:
    with open('done_vols_downsampled.json') as f:
        done = json.load(f)
except:
    done = []
    for vol in vols:
        if os.path.exists(os.path.join(vol, 'segmentation_new.lzma')):
            done.append(vol)

have_yet_tasks = [vol for vol in vols if vol not in done]

def downsample(vol_path):
    with lzma.open(os.path.join(vol_path, "segmentation.lzma")) as lzma_file:
        segmentation = np.frombuffer(lzma_file.read(),dtype=np.uint16).reshape(256,256,256)
    total_unique, total_counts = np.unique(segmentation, return_counts=True)
    total_counts_dict = dict(zip(total_unique, total_counts))

    downsampled_img = np.zeros((64,64,64), dtype=np.uint16)
    for i in range(64):
        for j in range(64):
            for k in range(64):
                subset = segmentation[i*4:i*4+4, j*4:j*4+4, k*4:k*4+4]
                unique, counts = np.unique(subset, return_counts=True)
                sub_counts = dict(zip(unique, counts))
                sorted_sub_counts = sorted(sub_counts.items(), key = lambda kv: kv[1], reverse=True)
                if len(sorted_sub_counts) == 1:
                    downsampled_img[i,j,k] = int(sorted_sub_counts[0][0])
                    continue
                max_count = sorted_sub_counts[0][1] if sorted_sub_counts[0][0] != 0 else sorted_sub_counts[1][1]
                ties = [item for item in sorted_sub_counts if item[1] == max_count and item[0] != 0]
                if len(ties) == 1:
                    downsampled_img[i,j,k] = int(ties[0][0])
                else:
                    total_counts_ties = sorted([(item[0], total_counts_dict[item[0]]) for item in ties], key = lambda kv: kv[1])
                    downsampled_img[i,j,k] = int(total_counts_ties[0][0])
    downsampled_img = downsampled_img.reshape((-1,)).astype(np.uint16).tobytes()
    with open(os.path.join(vol_path, "segmentation_new.lzma"),'wb') as f:
        with lzma.open(f, 'w') as lzf:
            lzf.write(downsampled_img)
    return


if __name__ == "__main__":
    for i, vol in tqdm(enumerate(have_yet_tasks), total=len(have_yet_tasks)):
        downsample(vol)
        done.append(vol)
        if (i+1) % CKPT == 0 or i+1 >= len(have_yet_tasks): 
            with open('done_vols_downsampled.json', 'w') as f:
                json.dump(done, f)
