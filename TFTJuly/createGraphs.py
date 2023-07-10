import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
state_results = []
for state in ['CA', 'CO', 'FL']:
    df = pd.read_csv('../statesJulySubmission/' + state + '/withArbovirusLocalizedMinMax.csv')
    wnvData = pd.read_csv('../statesJulySubmission/' + state + '/wnv_data.csv', index_col=[0])
    df.index = pd.to_datetime(df.time)
    wnvData.index = pd.to_datetime(wnvData.index)
    df = pd.concat([df, wnvData], axis=1)
    df = df.dropna()
    state_results.append(df)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 7))

ax1.plot(state_results[0]['500'], label='Prediction')
ax1.plot(state_results[0]['count'], label='Actual')
ax1.tick_params('x', labelrotation=90)
ax1.tick_params('y', labelsize=20)

ax2.plot(state_results[1]['500'], label='Prediction')
ax2.plot(state_results[1]['count'], label='Actual')
ax2.tick_params('x', labelrotation=90)
ax2.tick_params('y', labelsize=20)


ax3.plot(state_results[2]['500'], label='Prediction')
ax3.plot(state_results[2]['count'], label='Actual')
ax3.tick_params('x', labelrotation=90)
ax3.tick_params('y', labelsize=20)

lines_labels = [ax1.get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]


fig.legend(lines, labels, loc='upper center', ncol=2, fontsize=20)

plt.savefig('CaCoGa.png')
