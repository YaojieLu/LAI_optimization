
import pandas as pd
from MDP_class import MDP
from MDP_solver import value_iteration, policy_extraction
import timeit

# mdp class
mdp = MDP(gamma = 0.9, dL_unit = 0.2, gs_unit = 0.02, L_max = 10.0,
          mL = 2, cL = 10, mean_rd = 0.2)

## read utility from csv
#path = 'test/MDP_utility.csv'
#utility = read_utility(path)

# value iteration
# gap_s = 1000
start = timeit.default_timer()
utility1000 = value_iteration(mdp, gap_L = 5, gap_s = 1000, epsilon = 100, time_max = 20)
stop = timeit.default_timer()
print('Time to calculate utility: ', round((stop - start), 2))
# gap_s = 500
start = timeit.default_timer()
utility500 = value_iteration(mdp, gap_L = 5, gap_s = 500, epsilon = 10, time_max = 30,
                          utility = utility1000)
stop = timeit.default_timer()
print('Time to calculate utility: ', round((stop - start), 2))

# save utility
df_utility = pd.DataFrame.from_dict({(L, gs): utility500[L][gs]
                             for L in utility500.keys() 
                             for gs in utility500[L].keys()},
                            orient = 'index', columns = ['utility'])
df_utility.index = pd.MultiIndex.from_tuples(df_utility.index, names=['L', 's'])
df_utility = df_utility.reset_index()
df_utility.to_csv('test/MDP_utility.csv', sep = '\t')

# policy extraction
start = timeit.default_timer()
pi = policy_extraction(mdp, utility500, gap_L = 5, gap_s = 500)
stop = timeit.default_timer()
print('Time to calculate policy: ', round((stop - start), 2))

# save policy
df_policy = pd.DataFrame.from_dict({(L, gs): pi[L][gs]
                             for L in pi.keys() 
                             for gs in pi[L].keys()},
                            orient = 'index', columns = ['dL', 'gs', 'utility'])
df_policy.index = pd.MultiIndex.from_tuples(df_policy.index, names=['L', 's'])
df_policy = df_policy.reset_index()
df_policy.to_csv('test/MDP_policy.csv', sep = '\t')
