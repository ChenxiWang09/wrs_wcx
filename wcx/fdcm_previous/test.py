import numpy as np
date=np.load('objsize.npy', allow_pickle=True).item()
print(date[1])