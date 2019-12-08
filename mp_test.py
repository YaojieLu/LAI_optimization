
import numpy as np
import pandas as pd
from MDP_class import MDP
from scipy import interpolate
from multiprocessing import Pool
import timeit

# mdp class
mdp = MDP(gamma = 0.9, dL_unit = 0.2, gs_unit = 0.02, L_max = 2,
          mL = 2, cL = 10, mean_rd = 0.2)

### value iteration
# hyper parameters
gap_L = 5
gap_s = 500
epsilon = 100
time_max = 1

# class variables and methods
L_sp, s_sp, L_max, s_max = mdp.L_space, mdp.s_space, mdp.L_max, mdp.s_max
gamma, R, T, dL_sp, gs_sp = mdp.gamma, mdp.R, mdp.T, mdp.dL_space, mdp.gs_space

# create thinned state space
L_sp_td, s_sp_td = L_sp[0::gap_L], s_sp[0::gap_s]
if L_max not in L_sp_td: L_sp_td = np.append(L_sp_td, L_max)
if s_max not in s_sp_td: s_sp_td = np.append(s_sp_td, s_max)

# initialization
U_L = {s: 0.0 for s in s_sp}
U = {L: U_L for L in L_sp}

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

# multiprocessing step: Bellman update 
def bellman(L_array):
    U_td = []
    delta = 0
    # loop over L and s
    for L in L_array:
        U_td_L = []
        for s in s_sp_td:
            u = dc_dL_u(L, s)
            U_td_L.append(u)
        
        # update delta for utility with given L
        U_L = np.array([U[L][s] for s in s_sp_td])
        delta = max(delta, max(abs(U_L - U_td_L)))
        
        # update utility matrix for given L
        U_td.append(U_td_L)
    
    return U_td, delta
    
if __name__ == "__main__":
    # value iteration
    time = 0
    while True:
        
        # Bellman update
        pool = Pool(1)
        start = timeit.default_timer()
        U_td, delta = pool.map(bellman, (L_sp_td, ))[0]
        stop = timeit.default_timer()
        print('Time: {} and error: {} after {} iterations'
              .format(round((stop - start), 2), delta, time))
        
        # interpolate 2d
        f = interpolate.interp2d(s_sp_td, L_sp_td, np.array(U_td), kind = 'linear')
        U_array = f(s_sp, L_sp)
        # update utility matrix
        U = {L: {s: U_array[n_L, n_s] for n_s, s in enumerate(s_sp)}
             for n_L, L in enumerate(L_sp)}
        
        # stop criterion
        time += 1
        if delta <= epsilon*(1.0 - gamma)/gamma or time >= time_max:
            utility = U
            break
    
    # save utility
    df_utility = pd.DataFrame.from_dict({(L, gs): utility[L][gs]
                                 for L in utility.keys() 
                                 for gs in utility[L].keys()},
                                orient = 'index', columns = ['utility'])
    df_utility.index = pd.MultiIndex.from_tuples(df_utility.index, names=['L', 's'])
    df_utility = df_utility.reset_index()
    df_utility.to_csv('test/MDP_utility_mp.csv', sep = '\t')
