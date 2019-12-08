
import pandas as pd
from MDP_class import MDP
from MDP_solver import policy_extraction, read_utility
import timeit

# mdp class
mdp = MDP(gamma = 0.9, dL_unit = 0.1, gs_unit = 0.02, L_max = 10.0,
          mL = 2, cL = 10, mean_rd = 0.2)

# read utility from csv
path = 'Results/MDP_utility.csv'
utility = read_utility(path)

# policy extraction
start = timeit.default_timer()
pi = policy_extraction(mdp, utility, gap_L = 5, gap_s = 1000)
stop = timeit.default_timer()
print('Time to calculate policy: ', round((stop - start), 2))

# save policy
df_policy = pd.DataFrame.from_dict({(L, gs): pi[L][gs]
                             for L in pi.keys() 
                             for gs in pi[L].keys()},
                            orient = 'index', columns = ['dL', 'gs', 'utility'])
df_policy.index = pd.MultiIndex.from_tuples(df_policy.index, names=['L', 's'])
df_policy = df_policy.reset_index()
df_policy.to_csv('Results/MDP_policy.csv', sep = '\t')
