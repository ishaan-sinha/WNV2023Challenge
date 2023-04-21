import numpy as np
import pandas as pd


wnv_data = pd.read_csv('WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])
for state in [state for state in set(wnv_data['state']) if state != 'DC']:
    state_wnv = pd.read_csv('statesExtended/'+ state.strip() + '/wnv_data_' + state.strip() +'.csv', index_col=['year', 'month'])
    state_census = pd.read_csv('statesExtended/'+ state.strip() + '/census_' + state.strip() +'.csv', index_col=['year'])
    state_zindex = pd.read_csv('statesExtended/'+ state.strip() + '/' + state.strip() +'_zindex_data' + '.csv', index_col=['year', 'month'])
    state_precipitation = pd.read_csv('statesExtended/'+ state.strip() + '/' + state.strip() +'_precipitation_data' + '.csv', index_col=['year', 'month'])
    state_minTemp = pd.read_csv('statesExtended/'+ state.strip() + '/' + state.strip() +'_minTemp_data' + '.csv', index_col=['year', 'month'])
    state_maxTemp = pd.read_csv('statesExtended/'+ state.strip() + '/' + state.strip() +'_maxTemp_data' + '.csv', index_col=['year', 'month'])
    state_avgTemp = pd.read_csv('statesExtended/'+ state.strip() + '/' + state.strip() +'_avgTemp_data' + '.csv', index_col=['year', 'month'])
    final = state_wnv.copy()
    final['avgTemp'] = state_avgTemp['Value']
    final['minTemp'] = state_minTemp['Value']
    final['maxTemp'] = state_maxTemp['Value']
    final['precipitation'] = state_precipitation['Value']
    final['zindex'] = state_zindex['Value']
    for column in [i for i in state_census.columns if i != 'GEO_ID']:
        final[column] = np.nan
        for ind in final.index:
            if(ind[0] in state_census.index):
                final.loc[ind, column] = state_census.loc[ind[0], column]
    final.to_csv('statesExtended/'+ state.strip() + '/dropna_final_' + state.strip() +'.csv')
    print(state)


