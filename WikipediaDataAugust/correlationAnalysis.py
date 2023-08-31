import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


wnv_data = pd.read_csv('../WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

allstates_wnv = pd.DataFrame(columns = [i for i in wnv_data['state'].unique() if i not in ['DC']])

for state in [i for i in wnv_data['state'].unique() if i not in ['DC']]:
    state_Data = pd.read_csv('../statesJulySubmission/' + state + '/wnv_data.csv', index_col=[0])
    allstates_wnv[state] = state_Data['count']

files = ['wikishark-chart Avian influenza[en].csv',
'wikishark-chart Centers for Disease Control and Prevention[en].csv',
'wikishark-chart Common cold[en].csv',
'wikishark-chart Fever[en].csv',
'wikishark-chart Influenza[en].csv',
'wikishark-chart Main Page[en].csv',
'wikishark-chart Pandemic[en].csv',
'wikishark-chart West Nile virus[en].csv']

wiki_data = pd.DataFrame(columns = [i for i in files])
for i in files:
    wiki_data[i] = pd.read_csv('Monthly'+i, index_col=[0]).iloc[:,0]


wiki_data.index = [x[:-2]+'01' for x in wiki_data.index]
wiki_data.rename(
    {'wikishark-chart Avian influenza[en].csv': 'Avian influenza',
    'wikishark-chart Centers for Disease Control and Prevention[en].csv': 'Centers for Disease Control and Prevention',
    'wikishark-chart Common cold[en].csv': 'Common cold',
    'wikishark-chart Fever[en].csv': 'Fever',
    'wikishark-chart Influenza[en].csv': 'Influenza',
    'wikishark-chart Main Page[en].csv': 'Main Page',
    'wikishark-chart Pandemic[en].csv': 'Pandemic',
    'wikishark-chart West Nile virus[en].csv': 'West Nile virus'
     }, axis=1, inplace=True)

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