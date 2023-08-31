import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.ar_model import AutoReg


def getLogisticPrediction(state):
    def logifunc(x,l,c,k):
        return l / (1 + c*np.exp(-k*x))


    wnvData = pd.read_csv('../statesAugustSubmission/' + state + '/wnv_data.csv', index_col=[0])
    wnvData.index = pd.to_datetime(wnvData.index)
    yTotal = []

    optimal_params = []

    wnvData = wnvData[:-12]

    for year in wnvData['year'].unique():
        wnv_year = wnvData[wnvData.index.year == year]
        wnv_year_cumulative = wnv_year.cumsum()
        x = pd.array([i + 1 for i in range(len(wnv_year_cumulative))])
        y = wnv_year_cumulative['count'].values
        try:
            popt, pcov = curve_fit(logifunc, x, y, maxfev=2000)
        except:
            try:
                popt, pcov = curve_fit(logifunc, x, y, p0=optimal_params[-1], maxfev=2000)
            except:
                return 0

        y_pred = logifunc(x, *popt)
        y_pred[1:] -= y_pred[:-1]
        yTotal.append(y_pred)
        optimal_params.append(popt)

    p1 = [a[0] for a in optimal_params]
    p2 = [a[1] for a in optimal_params]
    p3 = [a[2] for a in optimal_params]

    model1 = AutoReg(p1, lags=1)
    model1_fit = model1.fit()
    p1.append(model1_fit.forecast(steps = 1)[0])

    model2 = AutoReg(p2, lags=1)
    model2_fit = model2.fit()
    p2.append(model2_fit.forecast(steps = 1)[0])

    model3 = AutoReg(p3, lags=1)
    model3_fit = model3.fit()
    p3.append(model3_fit.forecast(steps = 1)[0])

    total_result = []

    x = pd.array([i + 1 for i in range(12)])
    for i in range(len(p1)):
        summed = logifunc(x, p1[i], p2[i], p3[i])
        summed[1:] -= summed[:-1]
        for j in summed:
            total_result.append(j)
    return total_result