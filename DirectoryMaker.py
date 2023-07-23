import os
import pandas as pd

wnv_data = pd.read_csv('WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])


for i in set(wnv_data['state']):
    os.mkdir('statesAugustSubmission/'+i)

