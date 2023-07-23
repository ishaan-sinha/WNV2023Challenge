import pandas as pd
import numpy as np


wnv_data = pd.read_csv('WNV_forecasting_challenge_state-month_cases.csv')

wnv_data = wnv_data.groupby(['year', 'month'])['count'].sum()

wnv_data.index = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(wnv_data.index.get_level_values(0), wnv_data.index.get_level_values(1))])

pd.DataFrame(wnv_data).to_csv('national_count.csv')