import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
#
# Load AutoReg class from statsmodels.tsa.ar_model module
#
from statsmodels.tsa.ar_model import AutoReg
from sklearn.preprocessing import PowerTransformer
from statsmodels.tsa.statespace.sarimax import SARIMAX

wnv_data = pd.read_csv('WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

#for state in [state for state in set(wnv_data['state']) if state != 'DC']:
for state in ['CA']:
    state_data = pd.read_csv('statesExtended/'+ state.strip() + '/extended_final_' + state.strip() +'.csv', index_col=[0])
    state_data.dropna(inplace=True)
    state_data.index = pd.DatetimeIndex(state_data.index)
    state_data.index = pd.DatetimeIndex(state_data.index).to_period('M')

    state_data['1y'] = state_data['count'].shift(12, axis = 0)
    state_data['1m'] = state_data['count'].shift(1, axis = 0)
    state_data['2m'] = state_data['count'].shift(2, axis = 0)
    state_data['3m'] = state_data['count'].shift(3, axis = 0)
    state_data['3m_average'] = state_data[['1m', '2m', '3m']].mean(axis=1)
    state_data['diff1-2'] = state_data['1m'] - state_data['2m']
    state_data['diff2-3'] = state_data['2m'] - state_data['3m']
    state_data['month_sin'] = np.sin((state_data.index.month/12) * 2 * np.pi)
    state_data['month_cos'] = np.cos((state_data.index.month/12) * 2 * np.pi)
    state_data.dropna(inplace=True)

    cases = state_data['count']
    state_data.drop(['count'], axis=1, inplace=True)

    size = len(state_data)

    pt = PowerTransformer()
    print(state_data.columns)
    '''
    state_data = state_data[['avgTemp', 'minTemp', 'maxTemp', 'precipitation', 'zindex', 'household_income', '1y', '1m', '2m',
       '3m', '3m_average', 'diff1-2', 'diff2-3', 'month_sin', 'month_cos']]
       '''
    state_data = pt.fit_transform(state_data)

    train, test = cases[0:-12], cases[-12:]
    train_exog, test_exog = state_data[0:-12], state_data[-12:]
    aic_values = {}
    '''
    for p in range(1, 4):
        for q in range(1, 4):
            for P in range(1, 4):
                for Q in range(1, 4):
                    model = SARIMAX(train, order=(p, 0, q), seasonal_order=(P, 0, Q, 12),
                                    exog=train_exog)
                    model_fitted = model.fit(disp = 0)
                    aic_values[(p,q,P,Q)] = (model_fitted.aic)
        print("Checkpoint" + str(p))
    best_lag = min(aic_values, key=aic_values.get)
    print("best lag:" + str(best_lag))
    #Best lag is (1, 2,1,2)


    '''
    #(2, 2, 2)(1, 2, 1, 12) has r2 of 0.27
    model = SARIMAX(train, order=(2, 2, 2), seasonal_order=(1, 2, 1, 12),
                    exog=train_exog)
    model_fitted = model.fit()
    predictions = model_fitted.predict(start=len(train),
                                       end=len(train) + len(test) - 1,
                                       exog=test_exog, dynamic=False)

    compare_df = pd.concat([test, predictions], axis=1)
    
    #compare_df.to_csv('tempdf1.csv')
    plt.clf()
    figs, axes = plt.subplots(nrows=1, ncols=1)
    compare_df['count'].plot(ax=axes, label="actual")
    compare_df['predicted_mean'].plot(ax=axes, label="predicted")
    plt.suptitle("WNV Cases CA")
    plt.legend()
    plt.savefig('AllDataCASarimax')
    plt.show()

    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    print(mean_squared_error(compare_df['count'], compare_df['predicted_mean'], squared=False))
    print(mean_absolute_error(compare_df['count'], compare_df['predicted_mean']))
    print(r2_score(compare_df['count'], compare_df['predicted_mean']))






