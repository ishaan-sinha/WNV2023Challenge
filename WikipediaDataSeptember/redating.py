import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


wnv_data = pd.read_csv('../WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

wiki_data = pd.read_csv('MonthlywikiDataSeptember.csv', index_col=[0])

wiki_data.index = [x[:-2]+'01' for x in wiki_data.index]

wiki_data[:-1].to_csv('wiki_data.csv')
'''
for i in wiki_data.columns:
    wiki_data[i+'/mainpage'] = wiki_data[i]/wiki_data['Main Page']

combined_data = pd.concat([allstates_wnv, wiki_data], axis=1).dropna()
print(combined_data.columns)
corr_matrix = combined_data.corr()
fig, ax = plt.subplots(figsize=(30,30))
sns.heatmap(corr_matrix[30:], annot=False)

plt.show()
#plt.savefig('correlationMatrix.png')
'''