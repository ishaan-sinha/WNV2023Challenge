
import pandas as pd
import numpy as np

abundanceData = pd.read_csv('abundanceData.csv', index_col=['year', 'month'])
abundanceData.sort_index(inplace=True)

abundanceData.drop(columns=['Unnamed: 0', 'state', 'species'], inplace=True)


abundanceData = abundanceData.groupby(['year', 'month', 'statefp', 'countyfp'], sort=True)['rate/trap_night'].sum()
abundanceData.to_csv('combinedAbundanceData.csv')

df = pd.DataFrame(abundanceData)
df.reset_index(inplace=True)
df.set_index(['year', 'month'], inplace=True)
interpolated_df = pd.DataFrame(columns=['year', 'month' 'statefp', 'countyfp', 'min', 'max', 'mean', 'sum'])
for year_month in df.index:

    break