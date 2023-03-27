import pandas as pd
from darts import TimeSeries
from sklearn.preprocessing import PowerTransformer
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np


wnv_data = pd.read_csv('WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])


for state in [i for i in wnv_data['state'].unique() if i not in ['DC']]:
    print(state)
    state_data = pd.read_csv('states/' + state.strip() + '/extended_final_sarimax_' + state.strip() + '.csv', index_col=[0])
    state_data.dropna(inplace=True)
    state_data.index = pd.DatetimeIndex(state_data.index)
    state_data.index = pd.DatetimeIndex(state_data.index).to_period('M')

    state_data['1y'] = state_data['count'].shift(12, axis=0)
    state_data['1m'] = state_data['count'].shift(1, axis=0)
    state_data['2m'] = state_data['count'].shift(2, axis=0)
    state_data['3m'] = state_data['count'].shift(3, axis=0)
    state_data['3m_average'] = state_data[['1m', '2m', '3m']].mean(axis=1)
    state_data['diff1-2'] = state_data['1m'] - state_data['2m']
    state_data['diff2-3'] = state_data['2m'] - state_data['3m']
    state_data['month_sin'] = np.sin((state_data.index.month / 12) * 2 * np.pi)
    state_data['month_cos'] = np.cos((state_data.index.month / 12) * 2 * np.pi)
    state_data.dropna(inplace=True)

    cases = state_data['count']
    state_data.drop(['count'], axis=1, inplace=True)

    size = len(state_data)

    pt = PowerTransformer()
    #print(state_data.columns)
    '''
    state_data = state_data[
        ['avgTemp', 'minTemp', 'maxTemp', 'precipitation', 'zindex', 'household_income', '1y', '1m', '2m',
         '3m', '3m_average', 'diff1-2', 'diff2-3', 'month_sin', 'month_cos']]
         '''
    #print(state_data)
    #state_data = pt.fit_transform(state_data)

    train, test = cases[0:-12], cases[-12:]
    train_exog, test_exog = state_data[0:-12], state_data[-12:]

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
    '''
    plt.clf()
    figs, axes = plt.subplots(nrows=1, ncols=1)
    compare_df['count'].plot(ax=axes, label="actual")
    compare_df['predicted_mean'].plot(ax=axes, label="predicted")
    plt.suptitle("WNV Cases CA")
    plt.legend()
    plt.savefig('states/' + state.strip() + '/XGBoostonSarimaExtended_' + state.strip())
    plt.show()
    '''
    figs, axes = plt.subplots(nrows=1, ncols=1)
    total_compare['count'].plot(ax=axes, label="actual")
    total_compare['predicted_mean'].plot(ax=axes, label="predicted")
    plt.axvline(x=total_compare.index[-12], color='r', linestyle='--')
    plt.suptitle("WNV Cases CA")
    plt.legend()
    plt.savefig('states/' + state.strip() + '/FULLXGBoostonSarimaExtended_' + state.strip())
    plt.show()



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

