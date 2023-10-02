import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from census import Census
import us
from us import states
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


wnv_data = pd.read_csv('../WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])
c = Census("052c44076be6adb33959d3ee9e8b1c5db81760a6")

results = pd.DataFrame()
for state in set(wnv_data['state']):
    for year in range(2009, 2022):
        ca_census = c.acs5.state(fields=(
        'NAME', 'B01003_001E'), state_fips= eval("us.states."+state+".fips"), year=year)
        ca_census = pd.DataFrame(ca_census)
        population = ca_census['B01003_001E'][0]
        results = results.append({'state': state, 'year': year, 'population': population}, ignore_index=True)
results.to_csv('statePopulations.csv', index=False)
