import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error
from statsmodels.tools import eval_measures
import statsmodels.api as sm
import statsmodels.discrete.count_model as cm
from scipy.stats import nbinom


wnv_data = pd.read_csv('../WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

statsmodels_mae = pd.DataFrame(columns=['state', 'negative_binomial', 'poisson', 'zero_inflated_poisson','zero_inflated_negative_binomial'])

x = 7
quantiles = [0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600,
             0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990]

def get_negative_binomial(train):

    model = cm.NegativeBinomialP(endog=train.total_cases, exog=train.drop('total_cases', axis=1), p=1)
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
    state_data = pd.read_csv('../statesJuneSubmission/'+state+'/NOAA_data.csv')
    state_data.index = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(state_data.year, state_data.month)])
    wnvData = pd.read_csv('../statesJuneSubmission/' + state + '/wnv_data.csv', index_col=[0])
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
#for state in ['AL']:
    state_data = getData(state)
    state_data = state_data.dropna()
    state_data.index = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(state_data.year, state_data.month)])
    state_data.drop(['year', 'month'], axis = 1, inplace=True)


    state_data['total_cases'] = state_data['7monthsAhead']
    state_data.drop('7monthsAhead', axis=1, inplace=True)

    wnv_train = state_data[:-x]
    wnv_test = state_data[-x:]
    figs, axes = plt.subplots(nrows = 1, ncols = 1)

    results = [state]


    negative_binom_model = get_negative_binomial(wnv_train)
    temp = negative_binom_model.predict(exog = wnv_test.drop('total_cases', axis=1), which='model', y_values=quantiles)
    temp = pd.DataFrame(temp)

    temp[11].plot(label="Negative Binomial")
    state_data.total_cases[-x:].plot(label ="Actual")
    plt.legend()
    plt.savefig('../statesJulySubmission/' + state + '/test_negative_binomial.png')
    plt.clf()
    temp.to_csv('../statesJulySubmission/' + state + '/negative_binomial.csv')
    temp_1 = temp[11]
    

    poisson_model = get_poisson(wnv_train)
    temp = poisson_model.predict(exog = wnv_test.drop('total_cases', axis=1), which='model', y_values=quantiles)
    temp = pd.DataFrame(temp)

    temp[11].plot(label="Poisson")
    state_data.total_cases[-x:].plot(label ="Actual")
    plt.legend()
    plt.savefig('../statesJulySubmission/' + state + '/test_poisson.png')
    plt.clf()
    temp.to_csv('../statesJulySubmission/' + state + '/poisson.csv')
    temp_2 = temp[11]

    zero_inflated_poisson = get_zero_inflated_poisson(wnv_train)
    temp = zero_inflated_poisson.predict(exog = wnv_test.drop('total_cases', axis=1), which='prob', y_values=quantiles)
    temp = pd.DataFrame(temp)

    temp[11].plot(label="Zero Inflated Poisson")
    state_data.total_cases[-x:].plot(label ="Actual")
    plt.legend()
    plt.savefig('../statesJulySubmission/' + state + '/test_zero_inflated_poisson.png')
    plt.clf()
    temp.to_csv('../statesJulySubmission/' + state + '/zero_inflated_poisson.csv')
    temp_3 = temp[11]

    zero_inflated_negative_binomial = get_zero_inflated_negative_binomial(wnv_train)
    temp = zero_inflated_negative_binomial.predict(exog = wnv_test.drop('total_cases', axis=1), which='prob', y_values=quantiles)
    temp = pd.DataFrame(temp)

    temp[11].plot(label="Zero Inflated Negative Binomial")
    state_data.total_cases[-x:].plot(label ="Actual")
    plt.legend()
    plt.savefig('../statesJulySubmission/' + state + '/test_zero_inflated_negative_binomial.png')
    plt.clf()
    temp.to_csv('../statesJulySubmission/' + state + '/zero_inflated_negative_binomial.csv')
    temp_4 = temp[11]
    print(state)

    statsmodels_mae = statsmodels_mae.append({'state': state, 'negative_binomial': mean_absolute_error(temp_1, state_data.total_cases[-x:]), 'poisson': mean_absolute_error(temp_2, state_data.total_cases[-x:]), 'zero_inflated_poisson': mean_absolute_error(temp_3, state_data.total_cases[-x:]), 'zero_inflated_negative_binomial': mean_absolute_error(temp_4, state_data.total_cases[-x:])}, ignore_index=True)

statsmodels_mae.to_csv('../modelResults/July/statsmodels.csv')

