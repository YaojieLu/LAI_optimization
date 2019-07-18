
import pandas as pd
from MDP_class import MDP
from MDP_solver_old import value_iteration, expected_utility, read_utility
#import timeit
import matplotlib.pyplot as plt

# import utility
utility = read_utility('test/MDP_utility_convex_proof.csv')

# mdp class
mdp = MDP(gamma = 0.9, dL_unit = 0.2, gs_unit = 0.02, L_max = 10.0,
          mL = 2, cL = 10, mean_rd = 0.075)
gamma, R, T, dL_sp, gs_sp = mdp.gamma, mdp.R, mdp.T, mdp.dL_space, mdp.gs_space

## value iteration
#start = timeit.default_timer()
#utility = value_iteration(mdp, gap_L = 5, gap_s = 1000, time_max = 2)
#stop = timeit.default_timer()
#print('Time: ', round((stop - start), 2))

## save utility
#df_utility = pd.DataFrame.from_dict({(L, gs): utility[L][gs]
#                             for L in utility.keys() 
#                             for gs in utility[L].keys()},
#                            orient = 'index', columns = ['utility'])
#df_utility.index = pd.MultiIndex.from_tuples(df_utility.index, names=['L', 's'])
#df_utility = df_utility.reset_index()
#df_utility.to_csv('test/MDP_utility_convex_proof.csv', sep = '\t')

# expected_utility
L, s = 5, 0.5
df = pd.DataFrame([[round(dL, 2), gs, expected_utility(dL, gs, L, s, mdp, utility)]
     for dL in dL_sp(L) for gs in gs_sp(dL, L, s)], columns = ['dL', 'gs', 'utility'])

# figure
df2 = df[(df['gs'] % 0.1 ==0) & (df['dL'] % 1 == 0)]
fig = plt.figure(figsize = (8, 6))
ax1 = fig.add_subplot(121)
df2.groupby('dL').plot(x = "gs", y = "utility", ax = ax1)
ax2 = fig.add_subplot(122)
df2.groupby('gs').plot(x = "dL", y = "utility", ax = ax2)