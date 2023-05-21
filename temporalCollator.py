import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


wnv_data = pd.read_csv('WNVData/WNV_forecasting_challenge_state-month_cases.csv')

for state in [i for i in wnv_data['state'].unique() if i not in ['DC']]:
    state_data = wnv_data[wnv_data['state'] == state]
    state_data.index = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(state_data.year, state_data.month)])
    state_data.drop(['state', 'fips'], axis=1, inplace=True)
    state_data['1m'] = state_data['count'].shift(1)
    state_data['2m'] = state_data['count'].shift(2)
    state_data['3m'] = state_data['count'].shift(3)
    state_data['3mAverage'] = state_data[['1m', '2m', '3m']].mean(axis=1)
    state_data['12m'] = state_data['count'].shift(12)
    state_data['month_cos'] = np.cos(2 * np.pi * state_data.index.month / 12)
    state_data['month_sin'] = np.sin(2 * np.pi * state_data.index.month / 12)
    state_data.to_csv('statesMaySubmission/'+ state +'/temporalData.csv')
