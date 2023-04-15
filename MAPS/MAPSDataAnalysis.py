import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mapsData = pd.read_csv('IBP-MAPS-data-exploration-temporal_results.csv', index_col=0)
mapsData = mapsData[['Year', 'CommonName', 'N_individuals']]
mapsData = mapsData[mapsData['Year'] >= 2010]
newDf = pd.DataFrame(np.repeat(mapsData.values, 12, axis=0))
newDf['month'] = np.tile(np.arange(1, 13), len(mapsData))

newDf.to_csv('MAPS_data.csv')