import pandas as pd
from darts import TimeSeries
from sklearn.preprocessing import PowerTransformer
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np


wnv_data = pd.read_csv('WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

for state in [i for i in wnv_data['state'].unique() if i not in ['DC']]:
#for state in ['CA']:
    print(state)
    currentResults = pd.read_csv('statesExtended/' + state.strip() + '/scaledPredictions' + state.strip() + '.csv', index_col=0)

    state_data = pd.read_csv('statesExtended/' + state.strip() + '/withTemporalInputs_powerTransformed_' + state.strip() + '.csv',
                             index_col=[0])


    state_data.index = pd.DatetimeIndex(state_data.index)
    state_data.index = pd.DatetimeIndex(state_data.index).to_period('M')

    all_data = pd.read_csv('statesExtended/' + state.strip() + '/withAllInputs_' + state.strip() + '.csv', index_col=[0])
    cases = all_data['count']
    cases.index = pd.DatetimeIndex(cases.index)
    cases.index = pd.DatetimeIndex(cases.index).to_period('M')


    size = len(state_data)

    s = 24

    train, test = cases[0:-s], cases[-s:]
    train_exog, test_exog = state_data[0:-s], state_data[-s:]

    reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50)
    reg.fit(train_exog, train, eval_set=[(train_exog, train), (test_exog, test)], verbose=False)
    train_pred = pd.Series(reg.predict(train_exog))
    test_predictions = pd.Series(reg.predict(test_exog))
    total = pd.concat([train_pred, test_predictions])
    total.index = cases.index
    total_compare = pd.concat([cases, total], axis=1)
    total_compare.columns = ['count', 'predicted_mean']

    currentResults['XGBoost_temporal'] = total_compare['predicted_mean'].values
    currentResults.to_csv('statesExtended/' + state.strip() + '/scaledPredictions' + state.strip() + '.csv')
    '''
    test = test.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)
    compare_df = pd.concat([test, predictions], axis=1)
    compare_df.columns = ['count', 'predicted_mean']
    #print(compare_df)
    #print(total_compare)
    #compare_df.to_csv('statesExtended/' + state.strip() + '/XGBoostonSarimaExtended_data' + state.strip()+'.csv')

    plt.clf()
    figs, axes = plt.subplots(nrows=1, ncols=1)
    compare_df['count'].plot(ax=axes, label="actual")
    compare_df['predicted_mean'].plot(ax=axes, label="predicted")
    plt.suptitle("WNV Cases CA")
    plt.legend()
    #plt.savefig('statesExtended/' + state.strip() + '/XGBoostonSarimaExtended_' + state.strip())
    '''
    print(state)

