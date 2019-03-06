import lzma
import numpy as np
import os
import asyncio
from tqdm import tqdm

vols = [os.path.join('images', o) for o in os.listdir('images') if os.path.isdir(os.path.join('images', o))]

problems = []

def val_lzma(vol):
    problem = False
    try:
        with lzma.open(os.path.join(vol, "image.lzma")) as lzma_file:
            pixels = np.frombuffer(lzma_file.read(),dtype=np.uint8).reshape(64,64,64)
    except:
        problem = True
    return (vol, problem)

async def validate_lzma(vols):
    loop = asyncio.get_event_loop()
    futures = [
        loop.run_in_executor(
            None,
            val_lzma,
            vol
        )
        for vol in vols
    ]
    responses = [await f for f in tqdm(asyncio.as_completed(futures), total=len(futures))]
    for (vol, problem) in tqdm(responses):
        if problem:
            problems.append(vol)
    

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(validate_lzma(vols))

    print(problems)