import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from census import Census
from us import states
import os

wnv_data = pd.read_csv('WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])
c = Census("052c44076be6adb33959d3ee9e8b1c5db81760a6")


for state in set(wnv_data['state']):
    dataframes = []
    for year in range(2009, 2020):
        ca_census = c.acs5.state(fields=(
        'NAME', 'B01003_001E', 'B01001_002E', 'B01001_026E', 'B01001_003E', 'B01001_004E', 'B01001_005E', 'B01001_006E',
        'B01001_007E', 'B01001_008E', 'B01001_009E', 'B01001_010E', 'B01001_011E',
        'B01001_012E', 'B01001_013E', 'B01001_014E', 'B01001_015E', 'B01001_016E',
        'B01001_017E', 'B01001_018E', 'B01001_019E', 'B01001_020E', 'B01001_021E',
        'B01001_022E', 'B01001_023E', 'B01001_024E', 'B01001_025E','B01001_027E','B01001_028E','B01001_029E','B01001_030E','B01001_031E',
        'B01001_032E', 'B01001_033E', 'B01001_034E', 'B01001_035E', 'B01001_036E', 'B01001_037E', 'B01001_038E', 'B01001_039E',
        'B01001_040E', 'B01001_041E', 'B01001_042E', 'B01001_043E', 'B01001_044E', 'B01001_045E', 'B01001_046E', 'B01001_047E',
        'B01001_048E', 'B01001_049E', 'B02001_002E', 'B02001_003E', 'B02001_004E', 'B02001_005E', 'B02001_006E'), state_fips= eval("states."+state+".fips"), year=year)
        # B01003_001E = total population
        # B01001_002E = total male
        # B01001_026E = total female
        # B01001_003E = male under 5
        # B01001_004E = male 5 to 9
        # B01001_005E = male 10 to 14
        # B01001_006E = male 15 to 17
        # B01001_007E	Estimate!!Total:!!Male:!!18 and 19 years
        # B01001_008E	Estimate!!Total:!!Male:!!20 years
        # B01001_009E	Estimate!!Total:!!Male:!!21 years
        # B01001_010E	Estimate!!Total:!!Male:!!22 to 24 years
        # B01001_011E	Estimate!!Total:!!Male:!!25 to 29 years
        # B01001_012E	Estimate!!Total:!!Male:!!30 to 34 years
        # B01001_013E	Estimate!!Total:!!Male:!!35 to 39 years
        # B01001_014E	Estimate!!Total:!!Male:!!40 to 44 years
        # B01001_015E	Estimate!!Total:!!Male:!!45 to 49 years
        # B01001_016E	Estimate!!Total:!!Male:!!50 to 54 years
        # B01001_017E	Estimate!!Total:!!Male:!!55 to 59 years
        # B01001_018E	Estimate!!Total:!!Male:!!60 and 61 years
        # B01001_019E	Estimate!!Total:!!Male:!!62 to 64 years
        # B01001_020E	Estimate!!Total:!!Male:!!65 and 66 years
        # B01001_021E	Estimate!!Total:!!Male:!!67 to 69 years
        # B01001_022E	Estimate!!Total:!!Male:!!70 to 74 years
        # B01001_023E	Estimate!!Total:!!Male:!!75 to 79 years
        # B01001_024E	Estimate!!Total:!!Male:!!80 to 84 years
        # B01001_025E	Estimate!!Total:!!Male:!!85 years and over

        # B01001_027E	Estimate!!Total:!!Female:!!Under 5 years
        # B01001_028E	Estimate!!Total:!!Female:!!5 to 9 years
        # B01001_029E	Estimate!!Total:!!Female:!!10 to 14 years
        # B01001_030E	Estimate!!Total:!!Female:!!15 to 17 years
        # B01001_031E	Estimate!!Total:!!Female:!!18 and 19 years
        # B01001_032E	Estimate!!Total:!!Female:!!20 years
        # B01001_033E	Estimate!!Total:!!Female:!!21 years
        # B01001_034E	Estimate!!Total:!!Female:!!22 to 24 years
        # B01001_035E	Estimate!!Total:!!Female:!!25 to 29 years
        # B01001_036E	Estimate!!Total:!!Female:!!30 to 34 years
        # B01001_037E	Estimate!!Total:!!Female:!!35 to 39 years
        # B01001_038E	Estimate!!Total:!!Female:!!40 to 44 years
        # B01001_039E	Estimate!!Total:!!Female:!!45 to 49 years
        # B01001_040E	Estimate!!Total:!!Female:!!50 to 54 years
        # B01001_041E	Estimate!!Total:!!Female:!!55 to 59 years
        # B01001_042E	Estimate!!Total:!!Female:!!60 and 61 years
        # B01001_043E	Estimate!!Total:!!Female:!!62 to 64 years
        # B01001_044E	Estimate!!Total:!!Female:!!65 and 66 years
        # B01001_045E	Estimate!!Total:!!Female:!!67 to 69 years
        # B01001_046E	Estimate!!Total:!!Female:!!70 to 74 years
        # B01001_047E	Estimate!!Total:!!Female:!!75 to 79 years
        # B01001_048E	Estimate!!Total:!!Female:!!80 to 84 years
        # B01001_049E	Estimate!!Total:!!Female:!!85 years and over
        # B02001_002E White alone
        # B02001_003E	Estimate!!Total:!!Black or African American alone
        # B02001_004E	Estimate!!Total:!!American Indian and Alaska Native alone
        # B02001_005E	Estimate!!Total:!!Asian alone
        # B02001_006E	Estimate!!Total:!!Native Hawaiian and Other Pacific Islander alone
        ca_df = pd.DataFrame(ca_census)
        dataframes.append(ca_df)
    print(dataframes)
    break
