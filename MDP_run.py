
import pandas as pd
from MDP_class import MDP
from MDP_solver import value_iteration, policy_extraction#, read_utility

mdp = MDP(gamma = 0.9, dL_unit = 0.2, gs_unit = 0.02, L_max = 10.0, mL = 2, cL = 20, mean_rd = 0.1)

## read utility from csv
#path = 'test/MDP_utility.csv'
#utility = read_utility(path)

# value iteration
utility = value_iteration(mdp, gap_L = 5, gap_s = 1000, epsilon = 100, time_max = 70)
#df = pd.DataFrame.from_dict({(L, gs): utility[L][gs]
#                             for L in utility.keys() 
#                             for gs in utility[L].keys()},
#                            orient = 'index', columns = ['utility'])
#df.index = pd.MultiIndex.from_tuples(df.index, names=['L', 's'])
#df = df.reset_index()
#df.to_csv('test/MDP_utility.csv', sep = '\t')

# policy extraction
pi = policy_extraction(mdp, utility, gap_L = 5, gap_s = 500)
  
# export utility and policy
df = pd.DataFrame.from_dict({(L, gs): pi[L][gs]
                             for L in pi.keys() 
                             for gs in pi[L].keys()},
                            orient = 'index', columns = ['dL', 'gs', 'utility'])
df.index = pd.MultiIndex.from_tuples(df.index, names=['L', 's'])
df = df.reset_index()
df.to_csv('test/MDP_policy.csv', sep = '\t')
