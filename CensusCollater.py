import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from census import Census
from us import states
import os

c = Census("052c44076be6adb33959d3ee9e8b1c5db81760a6")

ca_census = c.acs5.state(fields = ('NAME', 'B01003_001E', 'B01001_002E', 'B01001_026E'),state_fips = states.CA.fips, year = 2019)

#B01003_001E = total population
#B01001_002E = total male
#B01001_026E = total female


ca_df = pd. DataFrame(ca_census)

print(ca_df)
print(ca_df.shape)