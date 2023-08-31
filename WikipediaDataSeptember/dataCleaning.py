import pandas as pd
import numpy as np

files = ['wikiDataSeptember.csv']

for file in files:
    currentFile = pd.read_csv('raw/'+file)
    currentFile.index = pd.to_datetime(currentFile['DateTime'])
    currentFile.drop(['DateTime'], axis=1, inplace=True)
    currentFile = currentFile.groupby(pd.Grouper(freq='M')).sum()
    currentFile.to_csv('Monthly'+file, index=True)
