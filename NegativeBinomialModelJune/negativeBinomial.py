import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.tools import eval_measures
import statsmodels.api as sm

wnv_data = pd.read_csv('../WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])


def get_negative_binomial(train, test):
    model_formula = "total_cases ~ 1 + "\
                    "month"
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

def get_poisson(train, test):
    model_formula = "total_cases ~ 1 + " \
                    "month"

    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.Poisson())
    fitted_model = model.fit()
    return fitted_model




#for state in [i for i in wnv_data['state'].unique() if i not in ['DC']]:
for state in ['CA']:
    wnvData = pd.read_csv('../statesJulySubmission/'+state+'/wnv_data.csv', index_col=[0])
    wnvData.index = pd.to_datetime(wnvData.index) #276
    wnvData = wnvData[['month', 'count']]
    wnvData['total_cases'] = wnvData['count']
    wnvData.drop('count', axis = 1, inplace=True)

    wnv_train = wnvData[:-50]
    wnv_test = wnvData[-50:]
    figs, axes = plt.subplots(nrows = 1, ncols = 1)

    '''
    negative_binom_model = get_negative_binomial(wnv_train, wnv_test)
    wnvData['negative_binomial'] = negative_binom_model.fittedvalues
    wnvData.negative_binomial.plot(label="Negative Binomial")
    '''
    '''
    poisson_model = get_poisson(wnv_train, wnv_test)
    wnvData['poisson_model'] = poisson_model.fittedvalues
    wnvData.poisson_model.plot(label = "Poisson")
    '''

    wnvData.total_cases.plot(label = "Actual")
    plt.axvline(x = wnvData.index[-50])
    plt.suptitle("WNV Predicted vs Actual")
    plt.legend()
    plt.show()

