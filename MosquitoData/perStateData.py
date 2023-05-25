import pandas as pd
import numpy as np

wnv_data = pd.read_csv('../WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

# Read in the data
abundanceData = pd.read_csv('combinedAbundanceData.csv')
infectionData = pd.read_csv('combinedInfectionData.csv')
abundanceData.set_index(['year','month','statefp','countyfp'], inplace=True)
infectionData.set_index(['year','month','statefp','countyfp'], inplace=True)
combinedData = pd.concat([abundanceData, infectionData], axis=1)
state_coordinates = pd.read_csv('stateCoordinates.csv', index_col=[0])

for state in [i for i in wnv_data['state'].unique() if i not in ['DC']]:
    state_data = combinedData.loc[combinedData['state'] == state]
    mean = state_data.groupby(['year','month']).mean()
    max = state_data.groupby(['year','month']).max()
    min = state_data.groupby(['year','month']).min()
    state_data = pd.concat([mean, max, min], axis=1)
    state_data.interpolate(method='linear', inplace=True)
    state_data.fillna(0, inplace=True)
    state_data.to_csv('MonthlyMosquitoData' + state + '.csv')
    print(state)