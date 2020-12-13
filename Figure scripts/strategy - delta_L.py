
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read csv
col_names = ['L_p', 's', 'dL', 'gs', 'utility']
df = pd.read_csv('../Results/MDP_policy.csv', sep='\t', header=0, names=col_names)

# parameters
Amax, gs50, dt = 15, 0.034, 1
gamma, dL_unit, gs_unit, L_max, mL, cL, mean_rd, k = 0.9, 0.1, 0.02, 10, 2, 10, 0.2, 0.1

# dataset
df['L'] = df['L_p'] + df['dL']
df['dL_0_max'] = np.where(df['dL'] < 0, 0, df['dL'])
df['A'] = 10*df['L']/(df['L']+3)*Amax*df['gs']/(df['gs']+gs50)*dt-mL*df['L']*dt-cL*df['dL_0_max']
df['E'] = df['L']*df['gs']
df['dL'] = round(df['dL'], 1)
df2 = df[['L_p', 's', 'dL']]
table = pd.pivot_table(df, values='dL', index=['L_p'], columns=['s'])

# figures
sns.set(font_scale=1.5)
ax = sns.heatmap(table, cmap='PiYG', center=0,
                 cbar_kws={'label': 'Change in leaf area, $\it{ΔL}$'})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.xlabel("Current relative soil water content, $\it{s}$", fontsize=18)
plt.ylabel("Current leaf area, $\it{L}$", fontsize=18)
plt.savefig('../Figures/Figure deltaL.png', bbox_inches = 'tight')
