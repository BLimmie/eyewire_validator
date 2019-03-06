import numpy as np
import cv2
from tqdm import tqdm
import os
import json
import lzma

vols = [os.path.join('images', o) for o in os.listdir('images') if os.path.isdir(os.path.join('images', o))]

try:
    with open('done_vols_converted.json') as f:
        done = json.load(f)
except:
    done = []
    for path in tqdm(vols):
        if len(os.listdir(path)) == 1:
            done.append(path)
        else:
            break
have_yet_tasks = [vol for vol in vols if vol not in done]

def combine_volumes(vol_path):
    full_image = np.zeros((256,256,256))
    for i in range(256):
        i_image_path = os.path.join(vol_path, "{}.jpg".format(i))
        image_array = cv2.imread(i_image_path,0)
        full_image[i,:,:] = image_array
    full_image = full_image.reshape((-1,)).astype(np.uint8).tobytes()
    with open(os.path.join(vol_path,'image.lzma'),'wb') as f:
        with lzma.open(f, 'w') as lzf:
            lzf.write(full_image)
                
    for i in range(256):
        i_image_path = os.path.join(vol_path, "{}.jpg".format(i))
        os.remove(i_image_path)
    return vol_path

for vol in tqdm(have_yet_tasks):
    combine_volumes(vol)
    done.append(vol)
    with open('done_vols_converted.json', 'w') as f:
        json.dump(done, f)