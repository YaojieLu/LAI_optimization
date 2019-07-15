
import multiprocessing as mp
import numpy as np

pool = mp.Pool()
param_cores = np.array_split(param_values, 10, axis = 0)
param_augmented = [(p, VPD, tmax, sp) for p in param_cores]
Y_pooled = pool.map(evaluate_model, param_augmented)
Y = np.vstack(Y_pooled)
