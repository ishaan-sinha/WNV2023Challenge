import pandas as pd
import numpy as np


# Read in the data
abundanceData = pd.read_csv('combinedAbundanceData.csv')
infectionData = pd.read_csv('combinedInfectionData.csv')
abundanceData.set_index(['year','month','statefp','countyfp'], inplace=True)
infectionData.set_index(['year','month','statefp','countyfp'], inplace=True)


combinedData = pd.concat([abundanceData, infectionData], axis=1)

print(combinedData)
mean = combinedData.groupby(['year','month']).mean()
max = combinedData.groupby(['year','month']).max()
min = combinedData.groupby(['year','month']).min()
combinedData = pd.concat([mean, max, min], axis=1)
combinedData.interpolate(method='linear', inplace=True)
combinedData.fillna(0, inplace=True)
#combinedData.to_csv('MonthlyMosquitoData.csv')