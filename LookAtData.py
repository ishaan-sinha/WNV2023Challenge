import pandas as pd
import matplotlib.pyplot as plt

wnv_data = pd.read_csv('WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

wnv_data_NJ = wnv_data[wnv_data['state'] == 'NJ']

wnv_data_NJ.plot(kind='line', y='count', figsize=(15, 5))

#plt.show()

print(sorted(set(wnv_data['fips'])))