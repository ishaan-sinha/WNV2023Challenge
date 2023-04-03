import pandas as pd
import numpy as np

wnv_data = pd.read_csv('WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])
currentResults = pd.DataFrame()
currentResults.index = sorted(list(set(wnv_data.state)))
print(currentResults.index)

currentResults.to_csv('currentResultsMAE.csv')