import pandas as pd
import numpy as np

wnv_data = pd.read_csv('../WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

finalSubmission = pd.DataFrame(columns=['location', 'forecast_date', 'target_end_date', 'target', 'type', 'quantile', 'value'])
#for state in [i for i in wnv_data['state'].unique() if i not in ['DC']]:
for state in ['CA']:
    state_pred = pd.read_csv('statesNormal/'+ state + '/firstPred.csv', index_col=[0])
    state_pred.index = pd.to_datetime(state_pred.index)
    import calendar
    from datetime import datetime
    for ind in state_pred.index:
        for col in state_pred.columns:
            finalSubmission.append({'location': state, 'forecast_date': '2023-04-30', 'target_end_date': (str(ind.year) + '-' + f"{a:02}" + '-' + calendar.monthrange(ind.year, ind.month)[1]), 'target': calendar.month_name[ind.month] + " WNV neuroinvasive disease cases" , 'type': 'quantile', 'quantile': col/100, 'value': state_pred.loc[ind, col]}, ignore_index=True)
finalSubmission.to_csv('submissions/finalSubmission1.csv')
