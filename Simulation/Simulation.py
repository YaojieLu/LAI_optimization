
import numpy as np
import pandas as pd
np.random.seed(123)
from matplotlib import pyplot as plt

# import the optimal strategy
col_names = ['L_p', 's', 'dL', 'gs', 'utility']
# run #14
opt = pd.read_csv('../Results/MDP_policy_27.csv', sep='\t', header=0, names=col_names)

# model parameters
L0 = 8 # initial leaf area
s0 = 0.3 # initial soil moisture
slope = 0.05 # transpiration function multiplier
dt = 1 # simulation time step
k = 0.1 # rainfall frequency
mean_rd = 0.1 # mean rainfall depth
n = 1000 # the number of rainfall events
L_unit = 0.5 # unit leaf area increment
s_unit = 0.05 # unit soil moisture increment
Amax = 20
gs50 = 0.034
mL = 2
cL = 10

# initalization
L_list = [L0]
s_list = [s0]
dL_list, gs_list, E_list, Anet_list, A_list, cL_list = [], [], [], [], [], []

# transpiration function
def Ef(L, gs, slope=slope, dt=dt):
    return slope*L*gs*dt

# generate a time series of rainfall
LD = np.random.exponential(scale=1/k, size=n) # dry periods
DR = np.floor(np.cumsum(LD)) # rainfall days
DR = DR.astype(int)
LS = DR[-1] # total simulation period - 1
AR = np.random.exponential(scale=mean_rd, size=n) # rainfall depths
AD = [0]*(LS+1) # the amount of rainfall on the ith day
for t in range(n):
    idx = int(DR[t])
    AD[idx] = AD[idx]+AR[t]

# slice function
def f1(L, s, action):
    return opt[(opt['L_p']==L) & (opt['s']==s)][action].values[0]
# 2d linear interpolation
def li2df(L, s, action, L_unit=L_unit, s_unit=s_unit):
    # slice function
    def f2(L, s):
        return f1(L, s, action=action)
    L0, s0 = round(int(L/L_unit)*L_unit, 3), min(1, round(int(s/s_unit)*s_unit, 3))
    L1, s1 = round(L0+L_unit, 3), min(1, round(s0+s_unit, 3))
    a00, a01, a10, a11 = f2(L0, s0), f2(L0, s1), f2(L1, s0), f2(L1, s1)
    if s0 == s1:
        print(s0, s1)
        a = (a00*(L1-L)+a10*(L-L0))/L_unit
    else:
        a = (a00*(L1-L)*(s1-s)+a10*(L-L0)*(s1-s)+a01*(L1-L)*(s-s0)+a11*(L-L0)*(s-s0))/(L_unit*s_unit)
    return a

# simulation
for t in range(LS):
    L_t, s_t = L_list[-1], s_list[-1]
    dL_t = li2df(L_t, s_t, 'dL')
    L_t = L_t+dL_t
    gs_t = li2df(L_t, s_t, 'gs')
    E_t = Ef(L_t, gs_t, slope=slope, dt=dt)
    R_t = AD[t]
    s_t = min(max(0, s_t-E_t+R_t), 1)
    A_t = L_t*10/(L_t+10)*Amax*gs_t/(gs_t+gs50)*dt
    cL_t = cL*max(0, dL_t)
    Anet_t = A_t - mL*L_t*dt - cL_t
    L_list.append(L_t)
    s_list.append(s_t)
    dL_list.append(dL_t)
    gs_list.append(gs_t)
    E_list.append(E_t)
    Anet_list.append(Anet_t)
    A_list.append(A_t)
    cL_list.append(cL_t)
results = [L_list, dL_list, gs_list, Anet_list, A_list, cL_list]

# figure
labels = ['$\it{L}$', '$\it{ΔL}$', '$\it{g_s}$', '$\it{A_net}$', '$\it{A}$', '$\it{c_L}$']
fig, axs = plt.subplots(2, 3, figsize=(20, 30))
i = 0
for row in axs:
    for col in row:
        col.plot(results[i])
        if i >= 3:
            col.set_xlabel('Days', fontsize=30)
        col.set_ylabel(labels[i], fontsize=30)
        col.tick_params(axis='both', which='major', labelsize=25)
        i += 1
plt.subplots_adjust(wspace=0.3, hspace=0.2)
#plt.savefig('../Figures/Figure simulation.png', bbox_inches = 'tight')

# output csv
results = [L_list, s_list, dL_list, gs_list, Anet_list]
df = pd.DataFrame(results).T
df.columns = ['L', 's', 'dL', 'gs', 'Anet']
df.to_csv('Simulation_27.csv', sep = '\t')
