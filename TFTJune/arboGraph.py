import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wnv_data = pd.read_csv('../WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

mae_1 = pd.read_csv('../modelResults/June/baselineThirdPredTest.csv')
mae_2 = pd.read_csv('../modelResults/June/withArbovirusTest.csv')
mae_1.index = mae_1['state']
mae_1 = mae_1.round(decimals=2)
mae_2.index = mae_2['state']
mae_2 = mae_2.round(decimals=2)

for i in set([i for i in wnv_data['state'] if i != 'DC']):
#for i in ['CA']:
    third_result = pd.read_csv('../statesJuneSubmission/' + i + '/thirdPred.csv', index_col=[0])['50']
    arbo_result = pd.read_csv('../statesJuneSubmission/' + i + '/withArbovirus.csv', index_col=[0])['50']
    real_result = pd.read_csv('../statesJuneSubmission/' + i + '/wnv_data.csv', index_col=[0])['count']
    total = pd.concat([third_result, arbo_result, real_result], axis=1)
    total.dropna(inplace=True)
    total.columns = ['mayPred', 'arboPred', 'real']
    total.mayPred.plot()
    total.arboPred.plot()
    total.real.plot()
    plt.legend()
    plt.title(f'May Model MAE: ' + str(mae_1.at[i, 'baselineThirdPredTest']) + '         Arbovirus Model MAE: ' + str(mae_2.at[i, 'withArbovirus']))
    plt.savefig('../statesJuneSubmission/' + i + '/arboGraph.png')
    plt.clf()
    #plt.show()
    print(i)