import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import seaborn as sns

wnv_data = pd.read_csv('WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

for state in [state for state in set(wnv_data['state']) if state != 'DC']:
#for state in ['CA']:
    state_data = pd.read_csv('states/' + state.strip() + '/withAllInputs_' + state.strip() + '.csv', index_col=[0])
    state_data = state_data[['count','male_under_17', 'male_18_to_40', 'male_40_to_64', 'male_over_65',
       'female_under_17', 'female_18_to_40', 'female_40_to_64',
       'female_over_65', 'total_population', 'male_population',
       'female_population', 'white_population', 'black_population',
       'native_american_population', 'asian_population',
       'pacific_islander_population', 'household_income']]
    state_data.index = pd.DatetimeIndex(state_data.index)
    state_data.index = pd.DatetimeIndex(state_data.index).to_period('M')
    correlations = state_data.corr()
    heatMap = sns.heatmap(correlations, cmap='coolwarm', xticklabels = True, yticklabels = True)
    #plt.show()
    plt.title('Correlation Heatmap for ' + state)

    heatMap.figure.tight_layout()
    plt.savefig('states/' + state.strip() + '/heatmapYearData_' + state.strip() + '.png')
    plt.show()
    plt.clf()
    print(state)