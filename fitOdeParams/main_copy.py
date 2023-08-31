import matplotlib.pyplot as plt
import numpy as np
import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, Parameter, report_fit
import scipy
from scipy.integrate import odeint

#read data

CA_data = pd.read_csv('../statesAugustSubmission/CA/wnv_data.csv')[['year', 'month', 'count']]
CA_data_2015 = CA_data[CA_data['year'] == 2015][5:12] #only 7 months used, from May to December

print(CA_data_2015)

#params
#PI_M = variable
PI_B = 1000
PI_H = 30
#MU_M = variable
MU_B = 1/1000
MU_H = 1/(70*365) #Reciprocal of length of human life in days
q = 0.1
c = 0.1
b_1 = 0.09 #adjusts peak
b_2 = 0.09*(1-c*q)
BETA_1 = 0.16 #adjusts peak and end value
BETA_2 = 0.88 #adjusts peak
BETA_3 = 0.88 #adjusts peak
d_B = 5 * 10^-5 #no noticable difference
d_H = 5 * 10^-7 #no noticable difference
ALPHA = 1/14 #Reciprocal of Incubation Period in 1/Days, Shifts end value
DELTA = 1 #adjusts peak
TAU = 1/14 #No noticable difference

'''population counts:
M_u = Uninfected mosquitoes
M_i = Infected mosquitoes
B_u = Uninfected birds
B_i = Infected birds
S = Susceptible
E = Asymptomatically Infected
I = Symptomatically Infected
H = Hospitalized Patients
R = Recovered

N_M = M_u + M_i
N_B = B_u + B_i
N_H = S + E + I'''




def f(t, y, paras):
    # Current state

    M_u, M_i, B_u, B_i, S, E, I, H, R = y


    # Model parameters
    try:
        PI_M, MU_M= paras['PI_M'].value, paras['MU_M'].value
    except:
        PI_M, MU_M = paras['PI_M'], paras['MU_M']

    N_M = M_u + M_i
    N_B = B_u + B_i
    N_H = S + E + I


    du = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Evaluate differential equations
    du[0] = PI_M - (b_1 * BETA_1 * M_u * B_i) / N_B - MU_M * M_u  # uninfected mosquitoes
    du[1] = (b_1 * BETA_1 * M_u * B_i) / (N_B) - MU_M * M_i  # infected mosquitoes

    du[2] = PI_B - (b_1 * BETA_2 * M_i * B_u) / (N_B) - MU_B * B_u  # uninfected birds
    du[3] = (b_1 * BETA_2 * M_i * B_u) / (N_B) - MU_B * B_i - d_B * B_i  # infected birds

    du[4] = PI_H - (b_2 * BETA_3 * M_i * S) / (N_H) - MU_H * S  # susceptible humans
    du[5] = (b_2 * BETA_3 * M_i * S) / (N_H) - MU_H * E - ALPHA * E  # asymptomatically infected humans

    du[6] = ALPHA * E - MU_H * I - DELTA * I  # symptomatically infected humans

    du[7] = DELTA * I - TAU * H - MU_H * H - d_H * H  # hospitalized humans
    du[8] = TAU * H - MU_H * R  # recovered humans
    return du

def g(t, x0, paras):
    """
    Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0
    """
    x = odeint(f, x0, t, args=(paras,))
    return x

def residual(paras, t, data):
    """
    compute the residual between actual data and fitted data
    """

    x0 = (paras['M_u'].value, paras['M_i'].value, paras['B_u'].value, paras['B_i'].value, paras['S'].value, paras['E'].value, paras['I'].value, paras['H'].value, paras['R'].value)
    model = g(t, x0, paras)

    # you only have data for one of your variables
    infected_model = model[:, 6]
    error = []

    for i in range(len(data)):
        error.append(infected_model[30*i] - data[i])
    if(np.isnan(error).any()):
        print("error is nan")
    return np.array(error)

#set parameters and bounds
params = Parameters()
params.add('PI_M', value=2000, min=0, max=100000)
params.add('MU_M', value=0.1, min=0, max=1)

params.add('M_u', value=500000, min=0, max=1000000)
params.add('M_i', value=5000, min=0, max=1000000)
params.add('B_u', value=500000, min=0, max=1000000)
params.add('B_i', value=5000, min=0, max=1000000)
params.add('S', value=5000000, min=0, max=100000000)

params.add('E', value=0, vary=False)
params.add('I', value=0, vary=False)
params.add('H', value=0, vary=False)
params.add('R', value=0, vary=False)

#initial conditions
M_u, M_i, B_u, B_i, S, E, I, H, R = 500000.0, 5000.0, 500000.0, 5000.0, 5000000.0, 0, 0, 0, 0
y0 = [M_u, M_i, B_u, B_i, S, E, I, H, R]
y0 = np.array(y0, dtype=np.float64)

sol = scipy.integrate.solve_ivp(fun=f, t_span=[0, 20], y0=y0, args=(params,), dense_output=True)
print(sol.message)
exit()
t = np.linspace(0, 20, 100)
z= sol.sol(t)[6]
plt.plot(t, z.T)
plt.show()

exit()
ode15s = scipy.integrate.ode(f)
ode15s.set_integrator('vode', method='bdf', order=5, nsteps=3000)
x = ode15s.integrate(1000, y0)