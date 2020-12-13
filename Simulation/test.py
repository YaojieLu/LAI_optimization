
import numpy as np
import pandas as pd
np.random.seed(123)

# import the optimal strategy
col_names = ['L_p', 's', 'dL', 'gs', 'utility']
opt = pd.read_csv('../Results/MDP_policy.csv', sep='\t', header=0, names=col_names)

# model parameters
L0 = 1 # initial leaf area
s0 = 0.3 # initial soil moisture
slope = 0.05 # transpiration function multiplier
dt = 1 # simulation time step
k = 0.1 # rainfall frequency
MAP = 3000 # mean annual precipitation
nZ = 0.5 # soil depth
gamma = 1/((MAP/365/k)/1000)*nZ
n = 5 # the number of rainfall events
L_unit = 0.5 # unit leaf area increment
s_unit = 0.05 # unit soil moisture increment

# initalization
L_list = [L0]
s_list = [s0]
dL_list, gs_list, E_list = [], [], []

# transpiration function
def Ef(L, gs, slope=slope, dt=dt):
    return slope*L*gs*dt

# generate a time series of rainfall
LD = np.random.exponential(scale=1/k, size=n) # dry periods
DR = np.floor(np.cumsum(LD)) # rainfall days
LS = int(DR[-1]) # total simulation period
AR = np.random.exponential(scale=1/gamma, size=n) # rainfall depths
AD = [0]*LS # the amount of rainfall on the ith day
for t in range(n):
    idx = int(DR[t])
    print(idx)
    AD[idx] = AD[idx]+AR[t]