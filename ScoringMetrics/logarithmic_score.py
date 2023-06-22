import numpy as np
import statsmodels.api as sm
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import log_loss

'''
pred_quantiles = [0.1, 0.5, 0.9]
pred_x = stats.zscore(pred_quantiles)

pred_y = [2, 5, 8]

model = sm.GLM(pred_y, pred_x, family=sm.families.NegativeBinomial())

result = model.fit()

glm = result.__getattribute__('model')

scipyModel = glm.get_distribution(result.params)

print(scipyModel.ppf(.1))
'''

'''
testy = [5]
predictions = [0.2, 0.5, 0.8]

losses = [log_loss(testy, [y for x in range(len(testy))]) for y in predictions]
print(losses)
'''

import numpy as np

observed = [5]
quantiles = [0.2, 0.5, 0.8]

def logarithmic_score(observed, quantiles):
    # Fit quantile forecasts with a parametric negative binomial distribution
    r = np.mean(observed)

    log_scores = []
    for q in quantiles:
        predicted = q * r

        # Calculate the logarithmic score
        log_score = np.log(pmf(observed, r, predicted))

        # Assign a value of -10 to logarithmic scores below -10
        log_score = np.where(log_score < -10, -10, log_score)

        log_scores.append(log_score)

    return np.array(log_scores)


def pmf(k, r, p):
    # Probability mass function of the negative binomial distribution
    coeff = np.exp(gammaln(r + k) - gammaln(k + 1) - gammaln(r))
    pmf = coeff * (p ** r) * ((1 - p) ** k)

    return pmf


def gammaln(x):
    # Natural logarithm of the gamma function
    return np.log(np.abs(np.math.gamma(x)))
