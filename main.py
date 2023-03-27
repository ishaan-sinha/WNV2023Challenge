import os
import pandas as pd

wnv_data = pd.read_csv('WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

for state in [i for i in wnv_data['state'].unique() if i not in ['DC']]:
    os.remove('states/' + state.strip() + '/FULLXGBoostonSarimaExtended_data' + state.strip()+'.csv')
    print(state)