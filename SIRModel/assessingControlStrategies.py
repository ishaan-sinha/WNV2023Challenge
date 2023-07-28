import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

#params
PI_m = 2000
PI_b = 1000
PI_h = 30
MU_m = 0.1
MU_b = 1/1000
MU_h = 1/(70*365)
q = 0.2
c = 0.2
b_1 = 0.09
b_2 = 0.09*(1 - c*q)
BETA_1 = 0.16
BETA_2 = 0.88
BETA_3 = 0.88
d_B = 5 * 10^(-5)
d_H = 5 * 10^(-7)
ALPHA = 1/14
DELTA = 1
TAU = 1/14


#initial values
M_u = 500000
M_i = 5000
B_u = 500000
B_i = 5000
S = 5000000
E = 500
H = 500
I = 500
R = 500
N_M = M_u + M_i
N_B = B_u + B_i
N_H = S+E+I+H+R

def update(M_u, M_i, B_u, B_i, S, E, I, H, R, N_M, N_B, N_H):
    dMudt = PI_m - (b_1*BETA_1*M_u*B_i)/(N_B) - MU_m*M_u
    dMidt = (b_1*BETA_1*M_u*B_i)/(N_B) - MU_m*M_i

    dBudt = PI_b - (b_1*BETA_2*B_u*M_i)/(N_B) - MU_b*B_u
    dBidt = (b_1*BETA_2*B_u*M_i)/(N_B) - MU_b*B_i - d_B*B_i

    dSdt = PI_h - (b_2*BETA_3*S*M_i)/(N_H) - MU_h*S
    dEdt = (b_2*BETA_3*S*M_i)/(N_H) - MU_h*E - ALPHA*E
    dIdt = ALPHA*E - MU_h*I - DELTA*I
    dHdt = DELTA*I - MU_h*H - TAU*H - d_H*H
    dRdt = TAU*H - MU_h*R

    M_u += dMudt
    M_i += dMidt
    B_u += dBudt
    B_i += dBidt
    S += dSdt
    E += dEdt
    I += dIdt
    H += dHdt
    R += dRdt

    N_M = M_u + M_i
    N_B = B_u + B_i
    N_H = S+E+I+H+R

    return M_u, M_i, B_u, B_i, S, E, I, H, R, N_M, N_B, N_H

results = [I+E]
for t in range(50):
    M_u, M_i, B_u, B_i, S, E, I, H, R, N_M, N_B, N_H = update(M_u, M_i, B_u, B_i, S, E, I, H, R, N_M, N_B, N_H)
    results.append(E+I)

results_x = [x for x in range(len(results))]

print(len(results))
print(results)

plt.plot(results_x, results)
plt.show()