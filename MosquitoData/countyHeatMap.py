import pandas as pd
import plotly.figure_factory as ff

abundanceData = pd.read_csv('abundanceData.csv')

fips = []

for row in abundanceData.iterrows():
    fips.append(('%02d' % row[1]['statefp']) + ('%03d' % (row[1]['countyfp'])))
fips = list(set(fips))

values = [5 for i in range(len(fips))]

fig = ff.create_choropleth(fips=fips, values=values)

fig.layout.template = None
#fig.show()
print(set(abundanceData['state']))