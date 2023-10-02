import pandas
import matplotlib.pyplot as plt


# Read in the data
df = pandas.read_csv('NM_4variables_CA.csv')


prefix = "NM"
state = "CA"
# Plot the data
plt.plot(df.iloc[:,0])
plt.title('UnInfected Mosquito')
plt.savefig('NM_CAparameter1.png')
plt.clf()
plt.plot(df.iloc[:,1])
plt.title('Infected Mosquito')
plt.savefig('NM_CAparameter2.png')
plt.clf()
plt.plot(df.iloc[:,2])
plt.title('Uninfected Birds')
plt.savefig('NM_CAparameter3.png')
plt.clf()
plt.plot(df.iloc[:,3])
plt.title('Infected Birds')
plt.savefig('NM_CAparameter4.png')


'''
df = pandas.read_csv('4variables_CO.csv')

# Plot the data
plt.plot(df.iloc[:,0])
plt.title('Infected Mosquito')
plt.savefig('COparameter1.png')
plt.clf()
plt.plot(df.iloc[:,1])
plt.title('Uninfected Birds')
plt.savefig('COparameter2.png')
plt.clf()
plt.plot(df.iloc[:,2])
plt.title('Infected Birds')
plt.savefig('COparameter3.png')
plt.clf()
plt.plot(df.iloc[:,3])
plt.title('Mosquito Birth Rate')
plt.savefig('COparameter4.png')
plt.clf()
'''