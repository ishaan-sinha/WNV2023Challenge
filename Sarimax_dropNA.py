import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
#
# Load AutoReg class from statsmodels.tsa.ar_model module
#
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX

wnv_data = pd.read_csv('WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])


#for state in [state for state in set(wnv_data['state']) if state != 'DC']:
for state in ['CA']:
    currentResults = pd.read_csv('states/'+state.strip()+'/scaledPredictions'+state.strip()+'.csv', index_col=0)
    currentResults['XGBoost_census'] = 0

    state_data = pd.read_csv('states/' + state.strip() + '/withAgeInputs_powerTransformed_' + state.strip() + '.csv', index_col=[0])
    state_data2 = pd.read_csv('states/' + state.strip() + '/withRaceInputs_powerTransformed_' + state.strip() + '.csv', index_col=[0])
    state_data = pd.concat([state_data, state_data2], axis=1)

    state_data.index = pd.DatetimeIndex(state_data.index)
    state_data.index = pd.DatetimeIndex(state_data.index).to_period('M')

    all_data = pd.read_csv('states/' + state.strip() + '/withAllInputs_' + state.strip() + '.csv', index_col=[0])
    cases = all_data['count']
    cases.index = pd.DatetimeIndex(cases.index)
    cases.index = pd.DatetimeIndex(cases.index).to_period('M')

    size = len(state_data)
    testSize = 24
    train, test = cases[0:-testSize], cases[-testSize:]
    train_exog, test_exog = state_data[0:-testSize], state_data[-testSize:]

    model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12),
                    exog=train_exog)
    model_fitted = model.fit()
    predictions = model_fitted.predict(start=len(train),
                                       end=len(train) + len(test) - 1,
                                       exog=test_exog, dynamic=False)
    compare_df = pd.concat([test, predictions], axis=1)

    plt.clf()
    figs, axes = plt.subplots(nrows=1, ncols=1)
    compare_df['count'].plot(ax=axes, label="actual")
    compare_df['predicted_mean'].plot(ax=axes, label="predicted")
    plt.suptitle("WNV Cases vs. Actual Cases" + state)
    plt.legend()
    plt.show()

    from sklearn.metrics import r2_score, mean_absolute_error

    MAE = mean_absolute_error(compare_df['count'], compare_df['predicted_mean'])
    currentResults['count'] = cases
    currentResults['XGBoost_census'] = predictions
    print(state)
    currentResults.to_csv('states/'+state.strip()+'/scaledPredictions'+state.strip()+'.csv')
    break
#currentResults.to_csv('currentResultsMAE.csv')