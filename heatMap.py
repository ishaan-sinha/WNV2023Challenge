import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import seaborn as sns

wnv_data = pd.read_csv('WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

#for state in [state for state in set(wnv_data['state']) if state != 'DC']:
for state in ['AL']:
    state_data = pd.read_csv('states/' + state.strip() + '/withAllInputs_' + state.strip() + '.csv', index_col=[0])
    state_data.index = pd.DatetimeIndex(state_data.index)
    state_data.index = pd.DatetimeIndex(state_data.index).to_period('M')

    years = sorted([year for year in set(state_data.index.year)])
    yearlydfs = []
    for year in years:
        year_data = state_data[state_data.index.year == year]
        correlations = year_data.corr()
        correlations = correlations['count']
        yearlydfs.append((year, correlations))
    combined = pandas.DataFrame()
    for year, correlations in yearlydfs:
        combined[year] = correlations
    print(combined)
    heatMap = sns.heatmap(combined, cmap='coolwarm', xticklabels = True, yticklabels = True)
    #plt.show()
    plt.savefig('states/' + state.strip() + '/heatmap_' + state.strip() + '.png')