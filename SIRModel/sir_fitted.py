import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize


def getSIR(seasonValues):

    xdata = [i+1 for i in range(len(seasonValues))]
    ydata = seasonValues

    ydata = np.array(ydata, dtype=float)
    xdata = np.array(xdata, dtype=float)

    def sir_model(y, x, beta, gamma):
        S = -beta * y[0] * y[1] / N
        R = gamma * y[1]
        I = -(S + R)
        return S, I, R

    def fit_odeint(x, beta, gamma):
        return integrate.odeint(sir_model, (S0, I0, R0), x, args=(beta, gamma))[:, 1]

    N = 1.0
    I0 = ydata[0]
    S0 = N - I0
    R0 = 0.0

    popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)
    fitted = fit_odeint(xdata, *popt)

    plt.plot(xdata, ydata, 'o')
    plt.plot(xdata, fitted)
    plt.show()

wnvData = pd.read_csv('../statesAugustSubmission/' + 'CA' + '/wnv_data.csv', index_col=[0])
wnvData.index = pd.to_datetime(wnvData.index)
year = 2005
wnv_year = wnvData[wnvData.index.year == year]

'''
wnv_startingMay = wnv_year[wnv_year.index.month >= 5]
wnv_startingMay.reset_index(inplace=True)

wnv_startingMay = wnv_startingMay.set_index('index').resample('D').interpolate()
'''

wnv_year.reset_index(inplace=True)
wnv_year = wnv_year.set_index('index').resample('D').interpolate()

sir_results = getSIR(wnv_year['count'].values)

'''
print(sir_results)
plt.plot(wnv_year.index, wnv_year['count'])
plt.plot(wnv_year.index, sir_results[3])

plt.show()
'''