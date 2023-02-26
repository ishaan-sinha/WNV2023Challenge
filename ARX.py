import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt

wnv_data_NJ = pd.read_csv('wnv_data_NJ.csv', index_col=[0]).drop(['fips', 'state', 'year', 'month'], axis=1)
wnv_data_NJ.index = pd.DatetimeIndex(wnv_data_NJ.index).to_period('M')

wnv_data_NJ_train = wnv_data_NJ.head(len(wnv_data_NJ) - 50)
wnv_data_NJ_test = wnv_data_NJ.tail(50)


arx_model = AutoReg(wnv_data_NJ_train['count'], lags=2, exog=wnv_data_NJ_train.drop('count', axis=1))
arx_results = arx_model.fit()

predictions = arx_results.predict(start=len(wnv_data_NJ_train), end=len(wnv_data_NJ) - 1,
                                  exog_oos=wnv_data_NJ_test.drop(['count'], axis=1), dynamic=False)

compare_df = pd.concat([wnv_data_NJ_test['count'], predictions], axis=1).rename(columns={'count': 'actual', 0:'predicted'})

plt.clf()
figs, axes = plt.subplots(nrows=1, ncols=1)
compare_df.actual.plot(ax=axes, label="actual")
compare_df.predicted.plot(ax=axes, label="predicted")

plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()
plt.savefig('ARX(2)[temperature].png')
plt.show()
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

compare_df.dropna(inplace=True)
print(mean_squared_error(compare_df['actual'], compare_df['predicted'], squared=False))
print(mean_absolute_error(compare_df['actual'], compare_df['predicted']))
print(r2_score(compare_df['actual'], compare_df['predicted']))
