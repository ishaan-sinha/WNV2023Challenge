import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.tools import eval_measures
import statsmodels.api as sm
import statsmodels.discrete.count_model as cm
from scipy.stats import nbinom

wnv_data = pd.read_csv('../WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

statsmodels_mae = pd.DataFrame(
    columns=['state', 'negative_binomial', 'poisson', 'zero_inflated_negative_binomial', 'zero_inflated_poisson'])

x = 5
quantiles = [0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600,
             0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990]


def get_negative_binomial(train):
    '''
    model_formula = "total_cases ~ 1 + "
    for i in [col for col in train.columns if col != "total_cases"]:
        model_formula += str(i) + " + "
    model_formula = model_formula[:-3]
    grid = 10 ** np.arange(-8, -3, dtype=np.float64)

    best_alpha = []
    best_score = 1000

    for alpha in grid:
        model = smf.glm(formula=model_formula,
                        data=train,
                        family=sm.families.NegativeBinomial(alpha=alpha))

        results = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)

        if score < best_score:
            best_alpha = alpha
            best_score = score
    print('best alpha = ', best_alpha)
    print('best score = ', best_score)

    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))
    fitted_model = model.fit()
    return fitted_model
    '''

    # train = train.append(test)
    model = cm.NegativeBinomialP(endog=train.total_cases, exog=train.drop('total_cases', axis=1), p=1)
    # distribution = model.get_distribution()

    fitted_model = model.fit_regularized()
    return fitted_model


def get_poisson(train):
    '''
    model_formula = "total_cases ~ 1 + "
    for i in [col for col in train.columns if col != "total_cases"]:
        model_formula += str(i) + " + "
    model_formula = model_formula[:-3]

    full_dataset = pd.concat([train])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.Poisson())
    fitted_model = model.fit()
    return fitted_model
    '''
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
    state_data = pd.read_csv('../statesSeptemberSubmission/'+state+'/NOAA_data.csv')
    state_data.index = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(state_data.year, state_data.month)])
    wnvData = pd.read_csv('../statesSeptemberSubmission/' + state + '/wnv_data.csv', index_col=[0])
    wnvData.index = pd.to_datetime(wnvData.index)
    state_data['count'] = wnvData['count']
    state_data['month_cos'] = np.cos(state_data.index.month * 2 * np.pi / 12)
    state_data['month_sin'] = np.sin(state_data.index.month * 2 * np.pi / 12)
    state_data['real_month_cos'] = np.cos((state_data.index + np.timedelta64(5, 'M')).month * 2 * np.pi / 12)
    state_data['real_month_sin'] = np.sin((state_data.index + np.timedelta64(5, 'M')).month * 2 * np.pi / 12)

    state_data['5monthsAhead'] = state_data['count'].shift(-5)
    state_data['6monthsAgo/1yearbeforePred'] = state_data['count'].shift(7)
    state_data.drop(['count'], axis=1, inplace=True)

    national_count = pd.read_csv('../WNVData/national_count.csv', index_col=[0]).iloc[:,0]
    national_count.index = pd.to_datetime(national_count.index)
    state_data['yearago_national_count'] = national_count
    state_data['yearago_national_count'] = state_data['yearago_national_count'].shift(7)
    return state_data


for state in [i for i in wnv_data['state'].unique() if i not in ['DC']]:
#for state in ['CA']:
    state_data = getData(state)
    state_data = state_data.dropna()
    state_data.index = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(state_data.year, state_data.month)])
    state_data.drop(['year', 'month'], axis=1, inplace=True)

    state_data['total_cases'] = state_data['5monthsAhead']
    state_data.drop('5monthsAhead', axis=1, inplace=True)

    wnv_train = state_data[:-x]
    wnv_test = state_data[-x:]
    figs, axes = plt.subplots(nrows=1, ncols=1)

    results = [state]


    negative_binom_model = get_negative_binomial(wnv_train)
    poisson_model = get_poisson(wnv_train)
    zero_inflated_poisson = get_zero_inflated_poisson(wnv_train)


    state_data['negative_binomial'] = negative_binom_model.predict(state_data.drop('total_cases', axis=1))
    results.append(eval_measures.meanabs(state_data.negative_binomial[-x:], wnv_test.total_cases[-x:]))
    state_data.drop('negative_binomial', axis=1, inplace=True)


    state_data['poisson_model'] = poisson_model.predict(state_data.drop('total_cases', axis=1)).astype(int)
    results.append(eval_measures.meanabs(state_data.poisson_model[-x:], wnv_test.total_cases[-x:]))
    state_data.drop('poisson_model', axis=1, inplace=True)

    try:
        zero_inflated_negative_binomial = get_zero_inflated_negative_binomial(wnv_train)
        state_data['zero_inflated_negative_binomial'] = zero_inflated_negative_binomial.predict(state_data.drop('total_cases', axis=1))
        results.append(eval_measures.meanabs(state_data.zero_inflated_negative_binomial[-x:], wnv_test.total_cases[-x:]))
        state_data.drop('zero_inflated_negative_binomial', axis=1, inplace=True)
    except:
        results.append(100000000)

    state_data['zero_inflated_poisson'] = zero_inflated_poisson.predict(state_data.drop('total_cases', axis=1)).astype(int)
    results.append(eval_measures.meanabs(state_data.zero_inflated_poisson[-x:], wnv_test.total_cases[-x:]))
    state_data.drop('zero_inflated_poisson', axis=1, inplace=True)

    statsmodels_mae.loc[len(statsmodels_mae)] = results

statsmodels_mae.to_csv('../modelResults/September/statsmodels_mae.csv')

