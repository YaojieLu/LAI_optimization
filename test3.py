import numpy as np

x = np.arange(10)

l = len(x)
print(x[::int((l+1)/4)])