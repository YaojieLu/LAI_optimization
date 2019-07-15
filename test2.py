
import multiprocessing as mp
import numpy as np

pool = mp.Pool(4)
out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))
