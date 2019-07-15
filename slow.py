
import numpy as np
from scipy import interpolate
import multiprocessing as mp

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
    
    # optimization given L
    def opt_L_f(L):
        U_td_L = []
        # utility matrix for given L
        for s in s_sp_td:
            u = max(R(dL, gs, L) + gamma*sum(p*U[L][s1]
                                             for (p, s1) in T(dL, gs, L, s))
                    for dL in dL_sp(L) for gs in gs_sp(dL, L, s))
            U_td_L.append(u)
        
        # delta for utility with given L
        U_L = np.array([U[L][s] for s in s_sp_td])
        delta_L = max(abs(U_L - U_td_L))
        
        return U_td_L, delta_L

    time = 0
    while True:
        U_td = []
        list_delta = []
        delta = 0
        
        for L in L_sp_td:
            #print(L)
            U_td_L, delta_L = opt_L_f(L)
            U_td.append(U_td_L)
            list_delta.append(delta_L)
        
        # interpolate 2d
        f = interpolate.interp2d(s_sp_td, L_sp_td, np.array(U_td), kind = 'linear')
        U_array = f(s_sp, L_sp)
        # update utility matrix
        U = {L: {s: U_array[i, j] for j, s in enumerate(s_sp)}
             for i, L in enumerate(L_sp)}
        # update delta
        delta = max(delta, max(list_delta))
        
        # stop criterion
        time += 1
        print('The error is {} after {} iterations'.format(delta, time))
        if delta <= epsilon*(1.0 - gamma)/gamma or time >= time_max:
            return U
