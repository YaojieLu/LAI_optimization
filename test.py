import pandas as pd
from MDP_class import MDP
from test2 import value_iteration
import timeit

mdp = MDP(gamma = 0.9, dL_unit = 0.2, gs_unit = 0.04, L_max = 2.0)

# value iteration
start = timeit.default_timer()
utility = value_iteration(mdp, gap_L = 5, gap_s = 1000, time_max = 1)
stop = timeit.default_timer()
print('Time: ', round((stop - start), 2))
df = pd.DataFrame.from_dict({(L, gs): utility[L][gs]
                             for L in utility.keys() 
                             for gs in utility[L].keys()},
                            orient = 'index', columns = ['utility'])
df.index = pd.MultiIndex.from_tuples(df.index, names=['L', 's'])
df = df.reset_index()
df.to_csv('test/2.csv', sep = '\t')
