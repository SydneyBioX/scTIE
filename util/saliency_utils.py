import numpy as np
import random
import os

def read_cell_ids(root, file):
    cell_ids = np.genfromtxt(os.path.join(root, file), delimiter=',')
    cell_ids = cell_ids[1:].astype(np.uint) 
    return cell_ids


def subsample_cell(probs, cell_ids, prop=1):
    n = probs.shape[0]
    print(n)
    ids = random.sample(range(n), round(n * prop))
    ids.sort()
    print(len(ids))
    return probs[ids], cell_ids[ids]

