import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

abundanceData = pd.read_csv('abundance.csv')
abundanceData['rate/trap_night'] = abundanceData['num_adult_females_collected']/abundanceData['num_trap_nights']
abundanceData = abundanceData[['rate/trap_night', 'year', 'month', 'state', 'statefp', 'countyfp', 'species']]

infectionData = pd.read_csv('infection.csv')
infectionData['inf/pool'] = infectionData['num_pools_wnv']/infectionData['num_mosquitoes']
infectionData = infectionData[['state', 'statefp', 'countyfp', 'year', 'month', 'species', 'inf/pool']]

#print(infectionData.head())
#print(abundanceData.head())

abundanceData.to_csv('abundanceData.csv')
infectionData.to_csv('infectionData.csv')

#print(set(abundanceData['state']))