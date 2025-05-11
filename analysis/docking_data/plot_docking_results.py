import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../../docking_results/docking_results.csv")

# Find minimum docking score for each molecule
df_new = df.groupby('SMILE')
mins = df_new['TotalEnergy'].min().values

# Plot distribution of docking scores and save
fig, ax = plt.subplots(figsize=[5,3], dpi=300)
ax.hist(mins, bins=np.arange(np.min(mins), np.max(mins), 2),
        color='lightblue', edgecolor='k', lw=0.25)
ax.set_xlabel(r'Minimum Docking Score / kcal mol$^{-1}$')
ax.set_ylabel('Number of Molecules')
ax.spines[['right', 'top']].set_visible(False)
plt.tight_layout()
fig.savefig('./distr_of_min_docking_scores.png')
