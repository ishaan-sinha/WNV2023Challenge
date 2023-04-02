import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
#
# Load AutoReg class from statsmodels.tsa.ar_model module
#
from statsmodels.tsa.ar_model import AutoReg
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import seaborn as sns

wnv_data = pd.read_csv('WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

for state in [state for state in set(wnv_data['state']) if state != 'DC']:
    state_data = pd.read_csv('states/' + state.strip() + '/extended_final_' + state.strip() + '.csv', index_col=[0])
    state_data.dropna(inplace=True)
    state_data.index = pd.DatetimeIndex(state_data.index)
    state_data.index = pd.DatetimeIndex(state_data.index).to_period('M')

    state_data['1y'] = state_data['count'].shift(12, axis=0)
    state_data['1m'] = state_data['count'].shift(1, axis=0)
    state_data['2m'] = state_data['count'].shift(2, axis=0)
    state_data['3m'] = state_data['count'].shift(3, axis=0)
    state_data['3m_average'] = state_data[['1m', '2m', '3m']].mean(axis=1)
    state_data['diff1-2'] = state_data['1m'] - state_data['2m']
    state_data['diff2-3'] = state_data['2m'] - state_data['3m']
    state_data['month_sin'] = np.sin((state_data.index.month / 12) * 2 * np.pi)
    state_data['month_cos'] = np.cos((state_data.index.month / 12) * 2 * np.pi)
    state_data.dropna(inplace=True)
    state_data.to_csv('states/' + state.strip() + '/withAllInputs_' + state.strip() + '.csv')

    count = state_data['count']
    state_data.drop(columns=['count'], inplace=True)
    state_data2 = state_data.copy()
    minmax = MinMaxScaler(feature_range=(1, 2))
    pt = PowerTransformer(method='yeo-johnson')
    pipeline = Pipeline([('minmax', minmax), ('power', pt)])
    state_data = pipeline.fit_transform(state_data)
    combined = pd.concat([pd.DataFrame(state_data, index=count.index, columns=state_data2.columns), count], axis=1)
    combined.to_csv('states/' + state.strip() + '/withAllInputs_powerTransformed_' + state.strip() + '.csv')




    state_data = pd.read_csv('states/' + state.strip() + '/extended_final_' + state.strip() + '.csv', index_col=[0])
    state_data.dropna(inplace=True)
    state_data.index = pd.DatetimeIndex(state_data.index)
    state_data.index = pd.DatetimeIndex(state_data.index).to_period('M')

    state_data['1y'] = state_data['count'].shift(12, axis=0)
    state_data['1m'] = state_data['count'].shift(1, axis=0)
    state_data['2m'] = state_data['count'].shift(2, axis=0)
    state_data['3m'] = state_data['count'].shift(3, axis=0)
    state_data['3m_average'] = state_data[['1m', '2m', '3m']].mean(axis=1)
    state_data['diff1-2'] = state_data['1m'] - state_data['2m']
    state_data['diff2-3'] = state_data['2m'] - state_data['3m']
    state_data['month_sin'] = np.sin((state_data.index.month / 12) * 2 * np.pi)
    state_data['month_cos'] = np.cos((state_data.index.month / 12) * 2 * np.pi)
    state_data.dropna(inplace=True)
    state_data_weather = state_data[['count', 'avgTemp', 'minTemp', 'maxTemp', 'precipitation', 'zindex']]
    state_data_age = state_data[['male_under_17', 'male_18_to_40', 'male_40_to_64', 'male_over_65',
       'female_under_17', 'female_18_to_40', 'female_40_to_64',
       'female_over_65']]
    state_data_race = state_data[['total_population', 'male_population',
       'female_population', 'white_population', 'black_population',
       'native_american_population', 'asian_population',
       'pacific_islander_population', 'household_income']]

    state_data_temporal = state_data[['1y','1m','2m','3m','3m_average','diff1-2','diff2-3','month_sin','month_cos']]
    state_data_weather.to_csv('states/' + state.strip() + '/withWeatherInputs_' + state.strip() + '.csv')
    state_data_age.to_csv('states/' + state.strip() + '/withAgeInputs_' + state.strip() + '.csv')
    state_data_race.to_csv('states/' + state.strip() + '/withRaceInputs_' + state.strip() + '.csv')
    state_data_temporal.to_csv('states/' + state.strip() + '/withTemporalInputs_' + state.strip() + '.csv')

    state_data_weather_powerTransformed = pd.DataFrame(pipeline.fit_transform(state_data_weather), index=state_data_weather.index, columns=state_data_weather.columns)
    state_data_age_powerTransformed = pd.DataFrame(pipeline.fit_transform(state_data_age), index=state_data_age.index, columns=state_data_age.columns)
    state_data_race_powerTransformed = pd.DataFrame(pipeline.fit_transform(state_data_race), index=state_data_race.index, columns=state_data_race.columns)
    state_data_temporal_powerTransformed = pd.DataFrame(pipeline.fit_transform(state_data_temporal), index=state_data_temporal.index, columns=state_data_temporal.columns)

    state_data_weather_powerTransformed.to_csv('states/' + state.strip() + '/withWeatherInputs_powerTransformed_' + state.strip() + '.csv')
    state_data_age_powerTransformed.to_csv('states/' + state.strip() + '/withAgeInputs_powerTransformed_' + state.strip() + '.csv')
    state_data_race_powerTransformed.to_csv('states/' + state.strip() + '/withRaceInputs_powerTransformed_' + state.strip() + '.csv')
    state_data_temporal_powerTransformed.to_csv('states/' + state.strip() + '/withTemporalInputs_powerTransformed_' + state.strip() + '.csv')

    print(state)