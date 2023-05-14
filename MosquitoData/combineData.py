
import pandas as pd
import numpy as np
'''
abundanceData = pd.read_csv('abundanceData.csv', index_col=['year', 'month'])
abundanceData.sort_index(inplace=True)

abundanceData.drop(columns=['Unnamed: 0', 'state', 'species'], inplace=True)


abundanceData = abundanceData.groupby(['year', 'month', 'statefp', 'countyfp'], sort=True)['rate/trap_night'].sum()
abundanceData.to_csv('combinedAbundanceData.csv')

df = pd.DataFrame(abundanceData)
df.reset_index(inplace=True)
df.set_index(['year', 'month'], inplace=True)
interpolated_df = pd.DataFrame(columns=['year', 'month' 'statefp', 'countyfp', 'min', 'max', 'mean', 'sum'])
'''

infectionData = pd.read_csv('infection.csv')
infectionData.drop(columns=['county', 'state', 'trap_type'], inplace=True)
infectionData = infectionData.groupby(['year', 'month', 'statefp', 'countyfp'], sort=True)
infectionData = infectionData.agg({'num_mosquitoes': 'sum', 'num_pools_wnv': 'sum'})

infectionData['infectionRate'] = infectionData['num_pools_wnv']/infectionData['num_mosquitoes']
infectionData.drop(columns=['num_mosquitoes', 'num_pools_wnv'], inplace=True)
infectionData.to_csv('combinedInfectionData.csv')
