import pandas as pd

from statsmodels.tsa.ar_model import AutoReg

# Read in the data
df = pd.read_csv('4variables_CO.csv')
true = df.iloc[-1]
df = df.iloc[:-1]
forecasts = {}
for column in df.columns:
    current = df[column]
    model = AutoReg(current, lags=1)
    result = model.fit()

    prediction = result.forecast(steps=1)
    forecasts[column] = prediction
print(pd.DataFrame(forecasts))
print(true)
