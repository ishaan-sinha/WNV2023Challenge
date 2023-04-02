import pandas as pd
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

wnv_data_NJ = pd.read_csv('states/CA/withAllInputs_powerTransformed_CA.csv', index_col=[0])
wnv_data_NJ.index = pd.DatetimeIndex(wnv_data_NJ.index).to_period('M')

wnv_data_NJ_train = wnv_data_NJ.head(len(wnv_data_NJ) - 50)
wnv_data_NJ_test = wnv_data_NJ.tail(50)

aic_values={}
'''
for p in range(1, 4):
    for q in range(1, 4):
        for P in range (1,3):
            for Q in range (1,3):
                model = SARIMAX(wnv_data_NJ_train['count'], order = (p, 0, q), seasonal_order=(P,0,Q,12), exog=wnv_data_NJ_train.drop('count', axis=1))
                model_fitted = model.fit()
                aic_values[(p,q,P,Q)] = (model_fitted.aic)
        print("Checkpoint" + str(P))
best_lag = min(aic_values, key=aic_values.get)
print("best lag:" + str(best_lag))
'''
#Best lag is (1,0,1)(1, 0, 1, 12)



model = SARIMAX(wnv_data_NJ_train['count'], order=(1, 0, 1), seasonal_order=(1, 0, 1, 12), exog=wnv_data_NJ_train.drop('count', axis=1))
model_fitted = model.fit()

predictions = model_fitted.predict(start=len(wnv_data_NJ_train), end=len(wnv_data_NJ) - 1, exog=wnv_data_NJ_test.drop(['count'], axis=1), dynamic=False)


compare_df = pd.concat([wnv_data_NJ_test['count'], predictions], axis=1).rename(columns={'count': 'actual', 'predicted_mean':'predicted'})

plt.clf()
figs, axes = plt.subplots(nrows=1, ncols=1)
compare_df.actual.plot(ax=axes, label="actual")
compare_df.predicted.plot(ax=axes, label="predicted")

plt.suptitle("WNV Predicted Cases vs. Actual Cases")
plt.legend()
#plt.savefig('SARIMAX(1,0,1)(1, 0, 1, 12)[temperature].png')
plt.show()
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

compare_df.dropna(inplace=True)
print(mean_squared_error(compare_df['actual'], compare_df['predicted'], squared=False))
print(mean_absolute_error(compare_df['actual'], compare_df['predicted']))
print(r2_score(compare_df['actual'], compare_df['predicted']))
