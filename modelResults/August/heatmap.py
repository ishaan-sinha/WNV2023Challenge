
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

baseline = pd.read_csv('baseline.csv', index_col=[0])
baselineWithNational = pd.read_csv('baselineWithNationalTest.csv', index_col=[0]).iloc[:, 1]
baselineWithNationalwithWiki = pd.read_csv('baselineWithNationalwithWikiTest.csv', index_col=[0]).iloc[:, 1]
statsmodels = pd.read_csv('statsmodels_mae.csv', index_col=[0]).iloc[:, [1,2,3,4]]
baselineWithNationalwithWikiwithLogistic = pd.read_csv('withArbovirusWithNationalwithWikiwithLogisticTest.csv', index_col=[0]).iloc[:, 1]

combined = pd.concat([baseline, baselineWithNational, baselineWithNationalwithWiki, statsmodels, baselineWithNationalwithWikiwithLogistic], axis=1)

combined = combined.set_index('state')


sns.heatmap(combined, vmin=0, vmax=5)

plt.xticks(rotation=20, fontsize=5)
plt.savefig('heatmap.png')