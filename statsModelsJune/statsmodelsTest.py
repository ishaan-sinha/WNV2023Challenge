import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.tools import eval_measures
import statsmodels.api as sm
import statsmodels.discrete.count_model as cm
from scipy.stats import nbinom

wnv_data = pd.read_csv('../WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

statsmodels_mae = pd.DataFrame(columns=['state', 'negative_binomial', 'zero_inflated_poisson'])

x = 12
quantiles = [0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600,
             0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990]


def get_negative_binomial(train):

    # train = train.append(test)
    model = cm.NegativeBinomialP(endog=train.total_cases, exog=train.drop('total_cases', axis=1), p=1)
    # distribution = model.get_distribution()
    try:
        fitted_model = model.fit()
    except:
        fitted_model = model.fit_regularized()
    return fitted_model


def get_poisson(train):
    model = cm.Poisson(endog=train.total_cases, exog=train.drop('total_cases', axis=1))
    fitted_model = model.fit_regularized()
    return fitted_model


def get_zero_inflated_negative_binomial(train):
    model = cm.ZeroInflatedNegativeBinomialP(endog=train.total_cases, exog=train.drop('total_cases', axis=1))
    fitted_model = model.fit_regularized()
    return fitted_model


def get_zero_inflated_poisson(train):
    model = cm.ZeroInflatedPoisson(endog=train.total_cases, exog=train.drop('total_cases', axis=1))
    fitted_model = model.fit_regularized()
    return fitted_model


def getData(state):
    state_data = pd.read_csv('../statesJulySubmission/' + state + '/NOAA_data.csv')
    state_data.index = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(state_data.year, state_data.month)])
    wnvData = pd.read_csv('../statesJulySubmission/' + state + '/wnv_data.csv', index_col=[0])
    wnvData.index = pd.to_datetime(wnvData.index)
    state_data['count'] = wnvData['count']

    state_data['month_cos'] = np.cos(state_data.index.month * 2 * np.pi / 12)
    state_data['month_sin'] = np.sin(state_data.index.month * 2 * np.pi / 12)

    state_data['real_month_cos'] = np.cos((state_data.index + np.timedelta64(7, 'M')).month * 2 * np.pi / 12)
    state_data['real_month_sin'] = np.sin((state_data.index + np.timedelta64(7, 'M')).month * 2 * np.pi / 12)

    state_data['7monthsAhead'] = state_data['count'].shift(-7)
    state_data['yearbeforePred'] = state_data['count'].shift(5)
    state_data.drop(['count'], axis=1, inplace=True)

    return state_data


for state in [i for i in wnv_data['state'].unique() if i not in ['DC']]:
#for state in ['CA']:
    state_data = getData(state)

    state_data.index = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(state_data.year, state_data.month)])
    state_data.drop(['year', 'month'], axis=1, inplace=True)

    state_data['total_cases'] = state_data['7monthsAhead']
    state_data.drop('7monthsAhead', axis=1, inplace=True)

    wnv_train = state_data[:-x].dropna()
    wnv_test = state_data[-x:].drop('total_cases', axis=1)
    figs, axes = plt.subplots(nrows=1, ncols=1)

    results = [state]
    '''
    negative_binom_model = get_negative_binomial(wnv_train)
    temp = negative_binom_model.predict(exog = wnv_test, which='model', y_values=quantiles)
    temp.set_axis([int(i * 1000) for i in quantiles], axis=1, inplace=True)
    temp = temp[-7:]
    temp.index = temp.index + pd.DateOffset(months=7)
    temp.to_csv('../statesJulySubmission/' + state + '/FORECASTnegative_binomial.csv')


    poisson_model = get_poisson(wnv_train)
    temp = poisson_model.predict(exog = wnv_test, which='model', y_values=quantiles)
    temp.set_axis([int(i * 1000) for i in quantiles], axis=1, inplace=True)
    temp = temp[-7:]
    temp.index = temp.index + pd.DateOffset(months=7)
    temp.to_csv('../statesJulySubmission/' + state + '/FORECASTpoisson.csv')
    '''

    zero_inflated_negative_binomial = get_zero_inflated_negative_binomial(wnv_train)
    temp = zero_inflated_negative_binomial.predict(exog = wnv_test, which='prob', y_values=quantiles)
    temp.set_axis([int(i * 1000) for i in quantiles], axis=1, inplace=True)
    temp = temp[-7:]
    temp.index = temp.index + pd.DateOffset(months=7)
    temp.to_csv('../statesJulySubmission/' + state + '/FORECASTzero_inflated_negative_binomial.csv')

    '''
    zero_inflated_poisson = get_zero_inflated_poisson(wnv_train)
    temp = zero_inflated_poisson.predict(exog=wnv_test, which='prob', y_values=quantiles)
    temp.set_axis([int(i * 1000) for i in quantiles], axis=1, inplace=True)
    temp = temp[-7:]
    temp.index = temp.index + pd.DateOffset(months=7)
    temp.to_csv('../statesJulySubmission/' + state + '/FORECASTzero_inflated_poisson.csv')
    '''
    print(state)




