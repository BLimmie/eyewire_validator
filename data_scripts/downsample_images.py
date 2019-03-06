from skimage.transform import downscale_local_mean
import numpy as np
import asyncio
import lzma
import os
from tqdm import tqdm

vols = [os.path.join('images', o) for o in os.listdir('images') if os.path.isdir(os.path.join('images', o))]

def downsample_vol_(vol_path):
    with lzma.open(os.path.join(vol_path, "image.lzma")) as lzma_file:
        image = np.frombuffer(lzma_file.read(),dtype=np.uint8)
    if image.size == 262144:
        return
    image = image.reshape(256,256,256)
    
    downsampled_img = np.rint(downscale_local_mean(image, (4,4,4)))
    downsampled_img = downsampled_img.reshape((-1,)).astype(np.uint8).tobytes()
    
    with open(os.path.join(vol_path,'image1.lzma'),'wb') as f:
        with lzma.open(f, 'w') as lzf:
            lzf.write(downsampled_img)
    os.remove(os.path.join(vol_path, "image.lzma"))
    os.rename(os.path.join(vol_path, "image1.lzma"), os.path.join(vol_path, "image.lzma"))
        

async def downsample_vol(vols):
    loop = asyncio.get_event_loop()
    futures = [
        loop.run_in_executor(
            None,
            downsample_vol_,
            vol
        )
        for vol in vols
    ]
    responses = [await f for f in tqdm(asyncio.as_completed(futures), total=len(futures))]
    
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(downsample_vol(vols))