
import numpy as np
import pandas as pd
from scipy import interpolate

def value_iteration(mdp, gap_L = 5, gap_s = 500,
                    epsilon = 100, time_max = 70, utility = None):
    """ Solving an MDP by value iteration """
    
    L_sp, s_sp, L_max, s_max = mdp.L_space, mdp.s_space, mdp.L_max, mdp.s_max
    gamma, R, T, dL_sp, gs_sp = mdp.gamma, mdp.R, mdp.T, mdp.dL_space, mdp.gs_space
    
    # create thinned state space
    L_sp_td, s_sp_td = L_sp[0::gap_L], s_sp[0::gap_s]
    if L_max not in L_sp_td: L_sp_td = np.append(L_sp_td, L_max)
    if s_max not in s_sp_td: s_sp_td = np.append(s_sp_td, s_max)
    
    # initialization
    U_L = {s: 0.0 for s in s_sp}
    U = utility or {L: U_L for L in L_sp}
    
    time = 0
    while True:
        U_td = []
        delta = 0
        
        for L in L_sp_td:
            #print(L)
            U_td_L = []
            for s in s_sp_td:
                u = max(R(dL, gs, L) + gamma*sum(p*U[L][s1]
                                                 for (p, s1) in T(dL, gs, L, s))
                        for dL in dL_sp(L) for gs in gs_sp(dL, L, s))
                U_td_L.append(u)
            
            # update delta for utility with given L
            U_L = np.array([U[L][s] for s in s_sp_td])
            delta = max(delta, max(abs(U_L - U_td_L)))
            
            # update utility matrix for given L
            U_td.append(U_td_L)
        
        # interpolate 2d
        f = interpolate.interp2d(s_sp_td, L_sp_td, np.array(U_td), kind = 'linear')
        U_array = f(s_sp, L_sp)
        # update utility matrix
        U = {L: {s: U_array[n_L, n_s] for n_s, s in enumerate(s_sp)}
             for n_L, L in enumerate(L_sp)}
        
        # stop criterion
        time += 1
        print('The error is {} after {} iterations'.format(delta, time))
        if delta <= epsilon*(1.0 - gamma)/gamma or time >= time_max:
            return U

def expected_utility(dL, gs, L, s, mdp, utility):
    """The expected utility of doing a in state s, according to  MDP and utility."""
    gamma, R, T = mdp.gamma, mdp.R, mdp.T
    return R(dL, gs, L) + gamma*sum(p*utility[L][s1] for (p, s1) in T(dL, gs, L, s))

def policy_extraction(mdp, utility, gap_L = 5, gap_s = 500):
    """Given an MDP and an utility function, determine the best policy,
    as a mapping from state to action."""
    pi = {}
    dL_sp, gs_sp = mdp.dL_space, mdp.gs_space
        
    # create thinned state space
    L_sp, s_sp, L_max, s_max = mdp.L_space, mdp.s_space, mdp.L_max, mdp.s_max
    L_sp_td, s_sp_td = L_sp[0::gap_L], s_sp[0::gap_s]
    if L_max not in L_sp_td: L_sp_td = np.append(L_sp_td, L_max)
    if s_max not in s_sp_td: s_sp_td = np.append(s_sp_td, s_max)
    
    for L in L_sp_td:
        print(L)
        pi_L = {}
        for count, s in enumerate(s_sp_td):
            def f(x):
                dL, gs = x
                return expected_utility(dL, gs, L, s, mdp, utility)
            pi_L[s] = max([(dL, gs) for dL in dL_sp(L) for gs in gs_sp(dL, L, s)],
                          key = f) + (utility[L][s], )
        pi[L] = pi_L
    
    return pi

# read utility from csv
def read_utility(path):
    df = pd.read_csv(path, sep = '\t', header = 0, index_col = ['L', 's'])
    d = {L: df.xs(L)['utility'].to_dict() for L in df.index.levels[0]}
    d = {L: {round(s, 6): v_s for s, v_s in v_L.items()} for L, v_L in d.items()}
    return d