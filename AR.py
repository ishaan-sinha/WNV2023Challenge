import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

wnv_data_NJ = pd.read_csv('wnv_data_NJ.csv')


wnv_data_NJ_train = wnv_data_NJ.head(len(wnv_data_NJ) - 50)
wnv_data_NJ_test = wnv_data_NJ.tail(50)
mse_values = {}

'''
#Find optimal Lag using function
from statsmodels.tsa.ar_model import ar_select_order
mod = ar_select_order(wnv_data_NJ_train['count'], maxlag=20)
print(mod.ar_lags)
#optimal lag is first 2
'''
'''
for lag in range(1, 21):
    model = AutoReg(wnv_data_NJ_train['count'], lags=lag)
    model_fitted = model.fit()
    predictions = model_fitted.predict(start = 0, end = len(wnv_data_NJ_train)-1, dynamic=False)
    compare_df = pd.concat([wnv_data_NJ_train['count'], predictions], axis=1).rename(
        columns={'count': 'actual', 0: 'predicted'})
    compare_df.dropna(inplace = True)
    mse = mean_absolute_error(compare_df['actual'], compare_df['predicted'])
    mse_values[lag] = mse
best_lag = min(mse_values, key= mse_values.get)
print("best lag:" + str(best_lag))
'''
#best lag is 12

model = AutoReg(wnv_data_NJ_train['count'], lags = 2)
model_fit = model.fit()
predictions = model_fit.predict(start = len(wnv_data_NJ_train), end = len(wnv_data_NJ)-1, dynamic=False)

compare_df = pd.concat([wnv_data_NJ_test['count'], predictions], axis=1).rename(columns={'count': 'actual', 0:'predicted'})

plt.clf()
figs, axes = plt.subplots(nrows=1, ncols=1)
compare_df.actual.plot(ax=axes, label="actual")
compare_df.predicted.plot(ax=axes, label="predicted")
plt.suptitle("WNV Predicted Cases vs. Actual Cases")
plt.legend()
plt.show()
#plt.savefig('AR(2).png')

print(mean_squared_error(compare_df['actual'], compare_df['predicted'], squared=False))
print(mean_absolute_error(compare_df['actual'], compare_df['predicted']))
print(r2_score(compare_df['actual'], compare_df['predicted']))
