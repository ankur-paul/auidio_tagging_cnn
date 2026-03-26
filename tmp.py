import time
from tqdm import tqdm

with tqdm(range(100), colour='blue', desc="Working", leave=False) as pbar:
    for i in pbar:
        time.sleep(0.1)
    pbar.set_postfix(loss=0.4, acc=0.9)
    pbar.colour = 'green'
    pbar.set_description("Done")
    pbar.refresh()
