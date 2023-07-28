import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numba
from scipy import integrate
from sklearn.metrics import mean_absolute_error as mae


def getSIR(seasonValues):
    N_choices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]

    I0 = seasonValues[0]+1
    RO = 0
    beta_choices = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

    gamma_choices = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

    tmax = 331
    Nt = 331
    t = np.linspace(0, tmax, Nt)

    choices_results = {}

    for N in N_choices:
        for beta in beta_choices:
            for gamma in gamma_choices:
                def derivative(X, t):
                    S, I, R = X
                    dotS = -beta * S * I / N
                    dotI = beta * S * I / N - gamma * I
                    dotR = gamma * I
                    return np.array([dotS, dotI, dotR])
                S0 = N - I0 - RO
                X0 = S0, I0, RO
                res = integrate.odeint(derivative, X0, t)
                S, I, R = res.T
                results = [I[0], I[30], I[60], I[90], I[120], I[150], I[180], I[210], I[240], I[270], I[300], I[330]]
                error = mae(seasonValues, results)
                choices_results[(N, beta, gamma)] = error
    (N_optimal, beta_optimal, gamma_optimal) = min(choices_results, key=choices_results.get)
    S0_optimal = N_optimal - I0 - RO
    X0_optimal = S0_optimal, I0, RO
    res_optimal = integrate.odeint(derivative, X0_optimal, t)
    S_optimal, I_optimal, R_optimal = res_optimal.T
    return (N_optimal, beta_optimal, gamma_optimal, [I_optimal[0], I_optimal[30], I_optimal[60], I_optimal[90], I_optimal[120], I_optimal[150], I_optimal[180], I_optimal[210], I_optimal[240], I_optimal[270], I_optimal[300], I_optimal[330]])


wnvData = pd.read_csv('../statesAugustSubmission/' + 'CA' + '/wnv_data.csv', index_col=[0])
wnvData.index = pd.to_datetime(wnvData.index)

year = 2005
wnv_year = wnvData[wnvData.index.year == year]

sir_results = getSIR(wnv_year['count'].values)
print(sir_results)
plt.plot(wnv_year.index, wnv_year['count'])
plt.plot(wnv_year.index, sir_results[3])

plt.show()



'''
N = 350. #Total number of individuals, N
I0, R0 = 1., 0 #Initial number of infected and recovered individuals
S0 = N - I0 - R0 #Susceptible individuals to infection initially is deduced
beta, gamma = 0.4, 0.1 #Contact rate and mean recovery rate
tmax = 160 #A grid of time points (in days)
Nt = 160
t = np.linspace(0, tmax, Nt+1)
def derivative(X, t):
    S, I, R = X
    dotS = -beta * S * I / N
    dotI = beta * S * I / N - gamma * I
    dotR = gamma * I
    return np.array([dotS, dotI, dotR])

X0 = S0, I0, R0 #Initial conditions vector
res = integrate.odeint(derivative, X0, t)
print(res)
S, I, R = res.T

plt.figure()
plt.grid()
plt.title("odeint method")
plt.plot(t, S, 'orange', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered with immunity')
plt.xlabel('Time t, [days]')
plt.ylabel('Numbers of individuals')
plt.ylim([0,N])
plt.legend()

plt.show();
'''