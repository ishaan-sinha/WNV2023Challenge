import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
#
# Load AutoReg class from statsmodels.tsa.ar_model module
#
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX

wnv_data = pd.read_csv('WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

#for state in [state for state in set(wnv_data['state']) if state != 'DC']:
for state in ['AL']:
    state_data = pd.read_csv('states/'+ state.strip() + '/dropna_final_' + state.strip() +'.csv')
    state_data.dropna(inplace=True)
    state_data.set_index(pd.to_datetime([f'{y}-{m}-01' for y, m in zip(state_data['year'], state_data['month'])]), inplace=True)
    state_data.index = pd.DatetimeIndex(state_data.index)
    state_data.index = pd.DatetimeIndex(state_data.index).to_period('M')
    state_data.sort_index(inplace = True)

    state_data['1y'] = state_data['count'].shift(12, axis = 0)
    state_data['1m'] = state_data['count'].shift(1, axis = 0)
    state_data['2m'] = state_data['count'].shift(2, axis = 0)
    state_data['3m'] = state_data['count'].shift(3, axis = 0)
    state_data['3m_average'] = state_data[['1m', '2m', '3m']].mean(axis=1)
    state_data['diff1-2'] = state_data['1m'] - state_data['2m']
    state_data['diff2-3'] = state_data['2m'] - state_data['3m']
    state_data['month_sin'] = np.sin(state_data['month']/(12 * 2 * np.pi))
    state_data['month_cos'] = np.cos(state_data['month']/(12 * 2 * np.pi))
    state_data.dropna(inplace=True)

    cases = state_data['count']
    state_data.drop(['Unnamed: 0', 'state', 'fips', 'count', 'year', 'month'], axis=1, inplace=True)

    size = len(state_data)
    train, test = cases[0:-1], cases[-1:]
    train_exog, test_exog = state_data[0:-1], state_data[-1:]

    model = SARIMAX(train, order=(2, 0, 2), seasonal_order=(1, 0, 1, 52),
                    exog=train_exog)
    model_fitted = model.fit()
    predictions = model_fitted.predict(start=len(train),
                                       end=len(train) + len(test) - 1,
                                       exog=test_exog, dynamic=False)
    compare_df = pd.concat([test, predictions], axis=1)

    compare_df.to_csv('tempdf1.csv')
    plt.clf()
    figs, axes = plt.subplots(nrows=1, ncols=1)
    compare_df.count.plot(ax=axes, label="actual")
    compare_df.predicted_mean.plot(ax=axes, label="predicted")
    plt.suptitle("WNV Cases vs. Actual Cases")
    plt.legend()
    plt.savefig('tempfig1')

    from sklearn.metrics import r2_score

    print(mean_squared_error(compare_df['count'], compare_df['predicted_mean'], squared=False))
    print(mean_absolute_error(compare_df['count'], compare_df['predicted_mean']))
    print(r2_score(compare_df['count'], compare_df['predicted_mean']))

    from sklearn.metrics import mean_squared_error, mean_absolute_error


