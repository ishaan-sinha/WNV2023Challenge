import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from shapely.geometry import box
from scipy.interpolate import griddata


df = pd.DataFrame('combinedAbundanceData.csv')
df.reset_index(inplace=True)
df.set_index(['year', 'month'], inplace=True)
interpolated_df = pd.DataFrame(columns=['year', 'month' 'statefp', 'countyfp', 'min', 'max', 'mean', 'sum'])
for year_month in df.index:

    break