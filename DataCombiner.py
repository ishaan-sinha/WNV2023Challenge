import pandas as pd


wnv_data = pd.read_csv('WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])
for state in set(wnv_data['state']):
    state_data = pd.read_csv('states/ '+ str(state) + '/wnv_data_' + str(state)+'.csv', index_col=['year', 'month'])
    print(state_data)


