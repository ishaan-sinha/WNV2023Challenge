import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

wnv_data = pd.read_csv('WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])


for state in [state for state in set(wnv_data['state']) if state != 'DC']:
    state_data = pd.read_csv('states/' + state.strip() + '/withAllInputs_powerTransformed_' + state.strip() + '.csv', index_col=[0])

    print(state)