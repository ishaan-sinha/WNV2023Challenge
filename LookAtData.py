import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

wnv_data = pd.read_csv('WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])
averageCase = {}
for i in set(wnv_data['state']):
    averageCase[i] = wnv_data[wnv_data['state'] == i]['count'].mean()
k = Counter(averageCase)

top3 = k.most_common(3)
print(top3)

'''
wnv_data_NJ = wnv_data[wnv_data['state'] == 'NJ']

wnv_data_NJ.plot(kind='line', y='count', figsize=(15, 5))

#plt.show()

print(sorted(set(wnv_data['fips'])))
'''