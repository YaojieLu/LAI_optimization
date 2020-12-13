
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_list = [('nyc', 'lax', 10,-10, -2,2),('nyc','chi',-9,9,-2,2),('nyc', 'sfo',-4,4,-1,1),
           ('nyc', 'mia', 2,-2, 2,-2),('lax','chi',5,-5,1,-1),('lax', 'sfo', -8,8, 2,-2),
           ('lax', 'mia', -6,6,0,0),('chi', 'sfo', 9,-9, -1,1),('chi', 'mia', 1,-1,3,-3),
           ('sfo', 'mia', -3,3, -2,2)]

df = pd.DataFrame(df_list, columns=['x', 'y','x-y','y-x','num1','num2'])

u = np.unique(df[["x","y"]].values)

p1 = df.pivot("y","x","x-y").reindex(u,u)
p2 = df.pivot("x","y","x-y").reindex(u,u) 
p = p1.combine_first(p2)
utri = np.triu(np.ones(p.shape)).astype(np.bool)
p.values[utri] = -p.values[utri]

n1 = df.pivot("y","x","num1").reindex(u,u)
n2 = df.pivot("x","y","num1").reindex(u,u) 
n = n1.combine_first(n2)
n.values[utri] = -n.values[utri]

# color according to n, labels according to p
ax = sns.heatmap(n, annot = p, center=0, cmap="RdBu")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()