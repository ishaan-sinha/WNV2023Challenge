import pandas as pd

wnv_data = pd.read_csv('WNVData/WNV_forecasting_challenge_state-month_cases.csv')


for i in set(wnv_data['state']):
#for i in ['DC']:
    wnv_data_state = wnv_data[wnv_data['state'] == i]
    wnv_data_state.index = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(wnv_data_state.year, wnv_data_state.month)]) #276 x 5
    wnv_data_state.index = pd.DatetimeIndex(wnv_data_state.index) #276 x 5
    wnv_data_state.to_csv('statesJulySubmission/'+i+'/wnv_data.csv')

'''
weather_data = pd.read_csv('AZTemperature.csv', index_col=['Date'])

wnv_data_NJ['temp'] = weather_data['Value'].values

wnv_data_NJ.to_csv('wnv_data_AZ.csv')
'''