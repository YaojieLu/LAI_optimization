
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# read csv
col_names = ['L_p', 's', 'dL', 'gs', 'utility']
df = pd.read_csv('../Results/MDP_policy.csv', sep='\t', header=0, names=col_names)

# parameters
Amax, gs50, dt = 15, 0.034, 1
gamma, dL_unit, gs_unit, L_max, mL, cL, mean_rd, k = 0.9, 0.1, 0.02, 10, 2, 10, 0.2, 0.1

# new columns
df['L'] = df['L_p'] + df['dL']
df['dL_0_max'] = np.where(df['dL'] < 0, 0, df['dL'])
df['A'] = 10*df['L']/(df['L']+3)*Amax*df['gs']/(df['gs']+gs50)*dt-mL*df['L']*dt-cL*df['dL_0_max']
df['E'] = df['L']*df['gs']
df['A/L'] = df['A']/df['L']

# figures
L_ps = [1, 5, 10]
fig, axs = plt.subplots(1, 1, figsize=(20, 20))
plt.show()

for L_p in L_ps:
    df_temp = df[df['L_p']==L_p]
    axs.plot(df_temp['s'], df_temp['dL'])
    axs.set_xlabel('$\it{s}$', fontsize=30)
    axs.set_ylabel('$\it{ΔL}$', fontsize=30)
    axs.tick_params(labelsize=25)
    axs.legend(labels=L_ps, loc='lower right', title='Existing leaf area',
                      fontsize='xx-large', title_fontsize='xx-large', markerscale=5)
plt.subplots_adjust(wspace=0.2, hspace=0.3)

