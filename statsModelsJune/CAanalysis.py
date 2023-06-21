import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.linalg as linalg
def getData(state):
    state_data = pd.read_csv('../statesJuneSubmission/' +state +'/NOAA_data.csv')
    state_data.index = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(state_data.year, state_data.month)])
    wnvData = pd.read_csv('../statesJuneSubmission/' + state + '/wnv_data.csv', index_col=[0])
    wnvData.index = pd.to_datetime(wnvData.index)
    state_data['count'] = wnvData['count']

    state_data['month_cos'] = np.cos(state_data.index.month * 2 * np.pi / 12)
    state_data['month_sin'] = np.sin(state_data.index.month * 2 * np.pi / 12)

    state_data['real_month_cos'] = np.cos((state_data.index + np.timedelta64(8, 'M')).month * 2 * np.pi / 12)
    state_data['real_month_sin'] = np.sin((state_data.index + np.timedelta64(8, 'M')).month * 2 * np.pi / 12)

    state_data['8monthsAhead'] = state_data['count'].shift(-8)
    state_data['yearbeforePred'] = state_data['count'].shift(4)
    state_data.drop(['count'], axis=1, inplace=True)

    return state_data

for i in ['CA']:
    state_data = getData(i)
    state_data = state_data.dropna()
    state_data.index = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(state_data.year, state_data.month)])
    state_data.drop(['year', 'month'], axis = 1, inplace=True)

    state_data['total_cases'] = state_data['8monthsAhead']
    state_data.drop('8monthsAhead', axis=1, inplace=True)

    wnv_train = state_data[:-8]
    wnv_test = state_data[-8:]
    figs, axes = plt.subplots(nrows = 1, ncols = 1)

    i = linalg.inv(wnv_train)

    print(i)
