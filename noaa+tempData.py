import numpy as np
import pandas as pd


wnv_data = pd.read_csv('WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

for state in [state for state in set(wnv_data['state']) if state != 'DC']:
    noaa = pd.read_csv('statesNormal/'+state + '/NOAA_data.csv')
    temporal = pd.read_csv('statesNormal/'+state + '/temporalData.csv')
    noaa.index = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(noaa.year, noaa.month)])
    noaa.drop(['year', 'month'], axis=1, inplace=True)
    temporal.index = pd.DatetimeIndex(temporal['Unnamed: 0'])
    temporal.drop(['Unnamed: 0'], axis=1, inplace=True)

    final = pd.concat([temporal, noaa], axis=1)
    final.to_csv('statesNormal/'+state + '/noaa+temporal.csv')
