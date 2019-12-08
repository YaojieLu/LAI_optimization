
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# read csv
col_names = ['L_p', 's', 'dL', 'gs', 'utility']
df = pd.read_csv('Results/MDP_policy.csv', sep='\t', header=0, names=col_names)

# parameters
Amax, gs50, dt = 15, 0.034, 1
gamma, dL_unit, gs_unit, L_max, mL, cL, mean_rd, k = 0.9, 0.1, 0.02, 10, 2, 10, 0.2, 0.1

# new columns
df['L'] = df['L_p'] + df['dL']
df['dL_0_max'] = np.where(df['dL'] < 0, 0, df['dL'])
df['A'] = 10*df['L']/(df['L']+3)*Amax*df['gs']/(df['gs']+gs50)*dt-mL*df['L']*dt-cL*df['dL_0_max']
df['E'] = df['L']*df['gs']

# figures
L_ps = [1, 5, 10]
ys = ['dL', 'gs']
labels = ['$\it{ΔL}$',
          '$\it{g_s}\ \mathrm{(mol\ m^{-2}\ s^{-1})}$']
#ys = ['gs', 'A', 'E']
#labels = ['$\it{g_s}\ \mathrm{(mol\ m^{-2}\ s^{-1})}$',
#          'Photosynthesis $\mathrm{(\mu mol\ m^{-2}\ s^{-1})}$',
#          'Transpiration $\mathrm{(mol\ m^{-2}\ s^{-1})}$']
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
plt.show()

i = 0
for ax in axs:
    for L_p in L_ps:
        df_temp = df[df['L_p']==L_p]
        ax.plot(df_temp['s'], df_temp[ys[i]])
        #ax.scatter(df_temp['s'], df_temp[ys[i]], s=5)
    ax.set_xlabel('$\it{s}$', fontsize=30)
    ax.set_ylabel(labels[i], fontsize=30)
    ax.tick_params(labelsize=25)
    i += 1
axs[0].legend(labels=L_ps, loc='lower right', title='Existing leaf area',
   fontsize='xx-large', title_fontsize='xx-large', markerscale=5)
#axs[1].set_title(title, fontsize=30)
plt.subplots_adjust(wspace=0.3)
plt.savefig('Figures/Figure 1.png', bbox_inches = 'tight')

