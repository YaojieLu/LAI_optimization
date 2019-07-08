
import pandas as pd
from MDP_class import MDP
from MDP_solver import value_iteration, policy_extraction#, read_utility

mdp = MDP(gamma = 0.9, dL_unit = 0.1, gs_unit = 0.01, L_max = 10.0, cL = 15)
print(mdp.L_space[::20])
print(mdp.s_space[::2000])

mdp = MDP(gamma = 0.9, dL_unit = 0.2, gs_unit = 0.02, L_max = 10.0, mL = 2, cL = 15)
print(mdp.L_space[::10])
print(mdp.s_space[::1000])
