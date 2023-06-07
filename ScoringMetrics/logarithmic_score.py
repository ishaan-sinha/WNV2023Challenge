import numpy as np
import statsmodels.api as sm
import pandas as pd
import scipy.stats as stats

pred_quantiles = [0.1, 0.5, 0.9]
pred_y = stats.zscore(pred_quantiles)

pred_x = [2, 5, 8]

model = sm.GLM(pred_y, pred_x, family=sm.families.NegativeBinomial())

result = model.fit()

fitted_values = result.fittedvalues

print(fitted_values)