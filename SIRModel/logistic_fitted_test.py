import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from logistic_fitted_forTesting import getLogisticPrediction
import matplotlib.pyplot as plt

wnv_data = pd.read_csv('../WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

results = pd.DataFrame(columns=['state', 'mae'])
for state in [i for i in wnv_data['state'].unique() if i not in ['DC']]:
#for state in ['CA']:
    data = pd.read_csv('../statesAugustSubmission/' + state + '/wnv_data.csv', index_col=[0])['count']
    data_train = data[:-12]
    data_test = data[-12:]
    test_fitted = getLogisticPrediction(state)
    if(test_fitted !=0):
        test_fitted_predictions = test_fitted[-12:]
        results.append({'state': state, 'mae': mae(data_test, test_fitted_predictions)}, ignore_index=True)
        plt.plot(data.values, label='Actual')
        plt.plot(test_fitted, label='Predicted')
        plt.legend()
        plt.savefig(state+'_logistic_fitted')
        plt.clf()

results.to_csv('logistic_fitted_results.csv')
