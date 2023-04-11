import pandas as pd
from darts import TimeSeries
from sklearn.preprocessing import PowerTransformer
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np


wnv_data = pd.read_csv('WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])
currentResults = pd.read_csv('currentResultsMAE.csv', index_col=0)
currentResults['XGBoost_weather+temporal'] = 0

for state in [i for i in wnv_data['state'].unique() if i not in ['DC']]:
#for state in ['NJ']:

    state_data = pd.read_csv('states/' + state.strip() + '/withWeatherInputs_powerTransformed_' + state.strip() + '.csv',
                             index_col=[0])
    state_data2 = pd.read_csv('states/' + state.strip() + '/withTemporalInputs_' + state.strip() + '.csv',index_col=[0])
    state_data = pd.concat([state_data, state_data2], axis=1)

    state_data.index = pd.DatetimeIndex(state_data.index)
    state_data.index = pd.DatetimeIndex(state_data.index).to_period('M')

    all_data = pd.read_csv('states/' + state.strip() + '/withAllInputs_' + state.strip() + '.csv', index_col=[0])
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
    predictions = pd.Series(reg.predict(test_exog))
    total = pd.concat([train_pred, predictions])
    total.index = cases.index
    total_compare = pd.concat([cases, total], axis=1)
    total_compare.columns = ['count', 'predicted_mean']
    test = test.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)
    compare_df = pd.concat([test, predictions], axis=1)
    compare_df.columns = ['count', 'predicted_mean']
    #print(compare_df)
    #print(total_compare)
    #compare_df.to_csv('states/' + state.strip() + '/XGBoostonSarimaExtended_data' + state.strip()+'.csv')

    plt.clf()
    figs, axes = plt.subplots(nrows=1, ncols=1)
    compare_df['count'].plot(ax=axes, label="actual")
    compare_df['predicted_mean'].plot(ax=axes, label="predicted")
    plt.suptitle("WNV Cases" +' ' + state)
    plt.legend()
    plt.savefig('states/' + state.strip() + '/XGBoostScaled_weather+temporal_' + state.strip())
    #plt.show()
    '''
    figs, axes = plt.subplots(nrows=1, ncols=1)
    total_compare['count'].plot(ax=axes, label="actual")
    total_compare['predicted_mean'].plot(ax=axes, label="predicted")
    plt.axvline(x=total_compare.index[-12], color='r', linestyle='--')
    plt.suptitle("WNV Cases CA")
    plt.legend()
    plt.savefig('states/' + state.strip() + '/FULLXGBoostonSarimaExtended_' + state.strip())
    plt.show()
    '''



    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    #print(mean_squared_error(compare_df['count'], compare_df['predicted_mean'], squared=False))
    #print(mean_absolute_error(compare_df['count'], compare_df['predicted_mean']))
    #print(r2_score(compare_df['count'], compare_df['predicted_mean']))
    '''
    print(mean_squared_error(total_compare['count'], total_compare['predicted_mean'], squared=False))
    print(mean_absolute_error(total_compare['count'], total_compare['predicted_mean']))
    print(r2_score(total_compare['count'], total_compare['predicted_mean']))
    '''
    MAE = mean_absolute_error(compare_df['count'], compare_df['predicted_mean'])
    currentResults['XGBoost_weather+temporal'][state] = MAE
    print(state)

currentResults.to_csv('currentResultsMAE.csv')
