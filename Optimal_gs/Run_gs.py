
import pandas as pd
from MDP_gs import *

mdp = MDP()

# value iteration
utility = fast_value_iteration(mdp, gap = 10)
#pi = best_policy(mdp, utility)
#pi_opt = policy_iteration(mdp)

# policy extraction
pi = best_policy(mdp, utility)

# export utility function
df = pd.DataFrame.from_dict(utility, orient = 'index', columns = ['utility'])
df.index.name = 's'
df = df.reset_index()
df['gs'] = df['s'].map(pi)
df.to_csv('Results/MDP.csv', sep = '\t')
