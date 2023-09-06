import pandas
import matplotlib.pyplot as plt


# Read in the data
df = pandas.read_csv('4variables_CA.csv')

# Plot the data
plt.plot(df.iloc[:,0])
plt.show()
plt.plot(df.iloc[:,1])
plt.show()
plt.plot(df.iloc[:,2])
plt.show()
plt.plot(df.iloc[:,3])
plt.show()