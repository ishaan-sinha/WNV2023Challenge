import pandas as pd
import numpy as np

# Read in the data
abundanceData = pd.read_csv('combinedAbundanceData.csv')
infectionData = pd.read_csv('combinedInfectionData.csv')
abundanceData.set_index(['year','month','statefp','countyfp'], inplace=True)
infectionData.set_index(['year','month','statefp','countyfp'], inplace=True)
combinedData = pd.concat([abundanceData, infectionData], axis=1)
combinedData = combinedData.groupby(['year','month']).mean()
combinedData.fillna(0, inplace=True)
combinedData.to_csv('averagedMonthlyMosquitoData.csv')