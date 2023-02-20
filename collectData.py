import pandas as pd

wnv_data = pd.read_csv('WNV_forecasting_challenge_state-month_cases.csv')
wnv_data_NJ = wnv_data[wnv_data['state'] == 'NJ']
wnv_data_NJ.index = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(wnv_data_NJ.year, wnv_data_NJ.month)]) #276 x 5
wnv_data_NJ.index = pd.DatetimeIndex(wnv_data_NJ.index).to_period('M') #276 x 5

weather_data = pd.read_csv('NJTemperature.csv', index_col=['Date'])

wnv_data_NJ['temp'] = weather_data['Value'].values

wnv_data_NJ.to_csv('wnv_data_NJ.csv')