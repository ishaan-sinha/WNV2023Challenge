import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in the data
df = pd.read_csv('combinedInfectionData.csv')
df = df.groupby(['statefp','countyfp'])
numberOfCounties = df.ngroups #100
#print('Number of counties: ', numberOfCounties)

keys = df.groups.keys()
sizes = [len(df.get_group(i)) for i in keys]
plt.hist(sizes, bins=20)
plt.savefig('infectionCountyGraphs/infectionSizes.png')
for i in keys:
    county_df = df.get_group(i)
    if(len(county_df) > 60):
        county_df['date'] = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(county_df.year, county_df.month)])
        county_df.plot('date', 'infectionRate', kind='scatter')
        plt.title('County: ' + str(i))
        plt.savefig('infectionCountyGraphs/' + str(i) + '.png')
        plt.clf()