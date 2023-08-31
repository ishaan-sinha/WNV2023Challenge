import pandas as pd
import numpy as np

files = ['wikishark-chart Avian influenza[en].csv',
'wikishark-chart Centers for Disease Control and Prevention[en].csv',
'wikishark-chart Common cold[en].csv',
'wikishark-chart Fever[en].csv',
'wikishark-chart Influenza[en].csv',
'wikishark-chart Main Page[en].csv',
'wikishark-chart Pandemic[en].csv',
'wikishark-chart West Nile virus[en].csv']

for file in files:
    currentFile = pd.read_csv('raw/'+file)
    currentFile.index = pd.to_datetime(currentFile['DateTime'])
    currentFile.drop(['DateTime'], axis=1, inplace=True)
    currentFile = currentFile.groupby(pd.Grouper(freq='M')).sum()
    currentFile.to_csv('Monthly'+file, index=True)
