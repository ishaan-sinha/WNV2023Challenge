import pandas as pd
import numpy as np
import datetime

from dateutil.relativedelta import relativedelta
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX

wnv_data = pd.read_csv('WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])
upTil = datetime.date(2023, 4, 1)

for state in [state for state in set(wnv_data['state']) if state != 'DC']:
    state_data = pd.read_csv('states/' + state.strip() + '/dropna_final_' + state.strip() + '.csv')
    state_data.set_index(pd.to_datetime([f'{y}-{m}-01' for y, m in zip(state_data['year'], state_data['month'])]), inplace=True)
    state_data.index = pd.DatetimeIndex(state_data.index)
    last_date = state_data.index[-1]
    indicesToAdd = pd.date_range(start=last_date + relativedelta(months=1), end=upTil, freq='MS')
    indicesToAdd = pd.DatetimeIndex(indicesToAdd)
    state_data = state_data.append(pd.DataFrame(index=indicesToAdd))
    state_data.index = pd.DatetimeIndex(state_data.index).to_period('M')
    state_data.sort_index(inplace=True)
    state_data.drop(columns=['Unnamed: 0', 'fips', 'state'], inplace=True)


    for column in [column for column in state_data.columns if column != 'year' and column != 'month']:
        currentSeries = state_data[column].dropna()
        currentEnd = datetime.date(currentSeries.index[-1].year, currentSeries.index[-1].month, 1)
        numberToForecast = (upTil.year - currentEnd.year) * 12 + (upTil.month - currentEnd.month)
        model = SARIMAX(currentSeries, order= (2,0,2), seasonal_order=(1, 0, 1, 12),  enforce_stationarity=False)
        model_fitted = model.fit()
        predictions = model_fitted.predict(start=len(currentSeries), end=len(currentSeries) + numberToForecast - 1, dynamic=False)
        predictions.index = pd.date_range(start=currentEnd + relativedelta(months=1), periods=len(predictions), freq='M')
        predictions.index = pd.DatetimeIndex(predictions.index).to_period('M')
        state_data.loc[predictions.index, column] = predictions
    state_data.drop(columns=['year', 'month'], inplace=True)
    state_data.to_csv('states/' + state.strip() + '/extended_final_sarimax_' + state.strip() + '.csv')
    print(state)

