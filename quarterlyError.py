import pandas as pd
from darts import TimeSeries
from sklearn.preprocessing import PowerTransformer
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np


wnv_data = pd.read_csv('WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])
random = pd.read_csv('currentResultsMAE.csv', index_col=[0])
currentResults = pd.DataFrame()
currentResults.index = random.index
currentResults['Q1'] = 0
currentResults['Q2'] = 0
currentResults['Q3'] = 0
currentResults['Q4'] = 0

for state in [i for i in wnv_data['state'].unique() if i not in ['DC']]:

    state_data = pd.read_csv('statesExtended/' + state.strip() + '/withWeatherInputs_powerTransformed_' + state.strip() + '.csv',
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
    predictions = pd.Series(reg.predict(test_exog))
    total = pd.concat([train_pred, predictions])
    total.index = cases.index
    total_compare = pd.concat([cases, total], axis=1)
    total_compare.columns = ['count', 'predicted_mean']
    test = test.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)
    compare_df = pd.concat([test, predictions], axis=1)
    compare_df.columns = ['count', 'predicted_mean']
    compare_df.index = cases.index[-s:]
    print(compare_df)
    #print(total_compare)
    #compare_df.to_csv('statesExtended/' + state.strip() + '/XGBoostonSarimaExtended_data' + state.strip()+'.csv')

    plt.clf()
    figs, axes = plt.subplots(nrows=1, ncols=1)
    compare_df['count'].plot(ax=axes, label="actual")
    compare_df['predicted_mean'].plot(ax=axes, label="predicted")
    plt.suptitle("WNV Cases CA")
    plt.legend()

    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    compare_df_Q1 = compare_df[compare_df.index.month.isin([1, 2, 3])]
    compare_df_Q2 = compare_df[compare_df.index.month.isin([4, 5, 6])]
    compare_df_Q3 = compare_df[compare_df.index.month.isin([7, 8, 9])]
    compare_df_Q4 = compare_df[compare_df.index.month.isin([10, 11, 12])]
    MAE1 = mean_absolute_error(compare_df_Q1['count'], compare_df_Q1['predicted_mean'])
    MAE2 = mean_absolute_error(compare_df_Q2['count'], compare_df_Q2['predicted_mean'])
    MAE3 = mean_absolute_error(compare_df_Q3['count'], compare_df_Q3['predicted_mean'])
    MAE4 = mean_absolute_error(compare_df_Q4['count'], compare_df_Q4['predicted_mean'])
    currentResults.loc[state, 'Q1'] = MAE1
    currentResults.loc[state, 'Q2'] = MAE2
    currentResults.loc[state, 'Q3'] = MAE3
    currentResults.loc[state, 'Q4'] = MAE4

    print(state)


currentResults.to_csv('QuarterlyXGBoostWeatherMAE.csv')
