
import numpy as np
import pandas as pd
from scipy import interpolate
from multiprocessing import Pool

def value_iteration(mdp, gap_L = 5, gap_s = 500,
                    epsilon = 100, time_max = 70, utility = None):
    """ Solving an MDP by value iteration """
    
    # class variables and methods
    L_sp, s_sp, L_max, s_max = mdp.L_space, mdp.s_space, mdp.L_max, mdp.s_max
    gamma, R, T, dL_sp, gs_sp = mdp.gamma, mdp.R, mdp.T, mdp.dL_space, mdp.gs_space
    
    # create thinned state space
    L_sp_td, s_sp_td = L_sp[0::gap_L], s_sp[0::gap_s]
    if L_max not in L_sp_td: L_sp_td = np.append(L_sp_td, L_max)
    if s_max not in s_sp_td: s_sp_td = np.append(s_sp_td, s_max)
    
    # initialization
    U_L = {s: 0.0 for s in s_sp}
    U = utility or {L: U_L for L in L_sp}
    
    ### divide and conquer algorithm
    # expected utility of doing a in state s
    def expected_utility(dL, gs, L, s):
        u = R(dL, gs, L) + gamma*sum(p*U[L][s1] for (p, s1) in T(dL, gs, L, s))
        return u
    # optimal gs given dL, L, and s
    def dc_gs_u(dL, L, s):
        
        # gs space
        arr = gs_sp(dL, L, s)
        
        # expected utility
        func = lambda gs :expected_utility(dL, gs, L, s)
        
        while True:
            len_arr = len(arr)
            
            if len_arr == 1:
                u = func(arr[0])
                return u
            elif len_arr == 2:
                x_l, x_r = arr[0], arr[1]
                u_l, u_r = func(x_l), func(x_r)
                u = max(u_l, u_r)
                return u
            else:
                mid = len_arr // 2
                x_l, x_m, x_r = arr[mid - 1], arr[mid], arr[mid + 1]
                u_l, u_m, u_r = func(x_l), func(x_m), func(x_r)
                if max(u_l, u_m, u_r) == u_m:
                    u = u_m
                    return u
                else:
                    arr = arr[: mid + 1] if u_l > u_r else arr[mid: ]
    # optimal dL given L and s
    def dc_dL_u(L, s):
        
        # dL space
        arr = dL_sp(L)
        
        # expected utility
        def func(dL):
            u = dc_gs_u(dL, L, s)
            return u
        
        while True:
            len_arr = len(arr)
            
            if len_arr == 1:
                u = func(arr[0])
                return u
            elif len_arr == 2:
                x_l, x_r = arr[0], arr[1]
                u_l, u_r = func(x_l), func(x_r)
                u = max(u_l, u_r)
                return u
            else:
                mid = len_arr // 2
                x_l, x_m, x_r = arr[mid - 1], arr[mid], arr[mid + 1]
                u_l, u_m, u_r = func(x_l), func(x_m), func(x_r)
                if max(u_l, u_m, u_r) == u_m:
                    u = u_m
                    return u
                else:
                    arr = arr[: mid + 1] if u_l > u_r else arr[mid: ]
    
    ### Bellman update
    time = 0
    while True:
        U_td = []
        delta = 0
        
        # loop over L and s
        for L in L_sp_td:
            print(L)
            U_td_L = []
            f = lambda s: dc_dL_u(L, s)
            pool = Pool()
            u = pool.map(f, s_sp_td)                
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

def policy_extraction(mdp, utility, gap_L = 5, gap_s = 500):
    """Given an MDP and an utility function, determine the best policy,
    as a mapping from state to action."""
    
    # initialization
    pi = {}
    
    # class variables
    gamma, R, T, dL_sp, gs_sp = mdp.gamma, mdp.R, mdp.T, mdp.dL_space, mdp.gs_space
    
    # create thinned state space
    L_sp, s_sp, L_max, s_max = mdp.L_space, mdp.s_space, mdp.L_max, mdp.s_max
    L_sp_td, s_sp_td = L_sp[0::gap_L], s_sp[0::gap_s]
    if L_max not in L_sp_td: L_sp_td = np.append(L_sp_td, L_max)
    if s_max not in s_sp_td: s_sp_td = np.append(s_sp_td, s_max)
    
    ### divide and conquer algorithm
    # expected utility of doing a in state s
    def expected_utility(dL, gs, L, s):
        u = R(dL, gs, L) + gamma*sum(p*utility[L][s1] for (p, s1) in T(dL, gs, L, s))
        return u
    # optimal gs given dL, L, and s
    def dc_gs(dL, L, s):
        
        # gs space
        arr = gs_sp(dL, L, s)
        
        # expected utility
        func = lambda gs :expected_utility(dL, gs, L, s)
        
        while True:
            len_arr = len(arr)
            
            if len_arr == 1:
                x_opt = arr[0]
                return x_opt
            elif len_arr == 2:
                x_l, x_r = arr[0], arr[1]
                u_l, u_r = func(x_l), func(x_r)
                x_opt = x_l if u_l > u_r else x_r
                return x_opt
            else:
                mid = len_arr // 2
                x_l, x_m, x_r = arr[mid - 1], arr[mid], arr[mid + 1]
                u_l, u_m, u_r = func(x_l), func(x_m), func(x_r)
                if max(u_l, u_m, u_r) == u_m:
                    x_opt = x_m
                    return x_opt
                else:
                    arr = arr[: mid + 1] if u_l > u_r else arr[mid: ]
    # optimal dL given L and s
    def dc_dL(L, s):
        
        # dL space
        arr = dL_sp(L)
        
        # expected utility
        def func(dL):
            gs_opt = dc_gs(dL, L, s)
            u = expected_utility(dL, gs_opt, L, s)
            return u
        
        while True:
            len_arr = len(arr)
            
            if len_arr == 1:
                x_opt = arr[0]
                return x_opt
            elif len_arr == 2:
                x_l, x_r = arr[0], arr[1]
                u_l, u_r = func(x_l), func(x_r)
                x_opt = x_l if u_l > u_r else x_r
                return x_opt
            else:
                mid = len_arr // 2
                x_l, x_m, x_r = arr[mid - 1], arr[mid], arr[mid + 1]
                u_l, u_m, u_r = func(x_l), func(x_m), func(x_r)
                if max(u_l, u_m, u_r) == u_m:
                    x_opt = x_m
                    return x_opt
                else:
                    arr = arr[: mid + 1] if u_l > u_r else arr[mid: ]
    
    # loop over L and s
    for L in L_sp_td:
        pi_L = {}
        for s in s_sp_td:
            dL_opt = dc_dL(L, s)
            gs_opt = dc_gs(dL_opt, L, s)
            pi_L[s] = (dL_opt, gs_opt) + (utility[L][s], )
        pi[L] = pi_L
    
    return pi

# read utility from csv
def read_utility(path):
    df = pd.read_csv(path, sep = '\t', header = 0, index_col = ['L', 's'])
    d = {L: df.xs(L)['utility'].to_dict() for L in df.index.levels[0]}
    d = {L: {round(s, 6): v_s for s, v_s in v_L.items()} for L, v_L in d.items()}
    return d