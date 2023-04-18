import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



EPOCHS = 400
INLEN = 32
HIDDEN = 64
LSTMLAYERS = 2
ATTHEADS = 1
DROPOUT = 0.1
BATCH = 32


N_FC = 24 #number of forecasts
RAND = 42           # set random state
N_SAMPLES = 100     # number of times a prediction is sampled from a probabilistic model
N_JOBS = -1
FIGSIZE = (9, 6)

QUANTILES = [0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99]
qL1, qL2, qL3 = 0.01, 0.05, 0.10        # percentiles of predictions: lower bounds
qU1, qU2, qU3 = 1-qL1, 1-qL2, 1-qL3     # upper bounds derived from lower bounds
label_q1 = f'{int(qU1 * 100)} / {int(qL1 * 100)} percentile band'
label_q2 = f'{int(qU2 * 100)} / {int(qL2 * 100)} percentile band'
label_q3 = f'{int(qU3 * 100)} / {int(qL3 * 100)} percentile band'

MSEAS = 60
ALPHA = 0.5


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel, NaiveSeasonal, NaiveDrift, ExponentialSmoothing
from darts.utils.statistics import check_seasonality, extract_trend_and_seasonality
from darts.metrics import mape, mae

from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.utils import ModelMode, SeasonalityMode, TrendMode

pd.set_option("display.precision",2)
np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = '{:,.2f}'.format

all_data = pd.read_csv('../states/CA/withAllInputs_CA.csv', index_col=[0])
cases = all_data['count']
cases.index = pd.DatetimeIndex(cases.index)
#cases.index = pd.DatetimeIndex(cases.index).to_period('M')

state_data = pd.read_csv('../states/CA/withWeatherInputs_CA.csv', index_col=[0])
state_data.index = pd.DatetimeIndex(state_data.index)
#state_data.index = pd.DatetimeIndex(state_data.index).to_period('M')

cases = TimeSeries.from_series(cases)
plt.figure(100, figsize=(18, 5))
cases.plot()
#plt.show()

# check for seasonality, via ACF
mseas=MSEAS
for m in range(2, MSEAS):
    is_seasonal, mseas = check_seasonality(cases, m=m, alpha=ALPHA, max_lag=MSEAS)
    if is_seasonal:
        break

print("seasonal? " + str(is_seasonal))
if is_seasonal:
    print('There is seasonality of order {} months'.format(mseas))
#seasonality of 12 months



# ETS analysis: try to discern trend and seasonal components
cases_trend, cases_seas = extract_trend_and_seasonality(ts=cases, freq=mseas, model = ModelMode.ADDITIVE)

plt.figure(100, figsize=(18, 5))
cases_trend.plot()
plt.title("trend component")
#plt.show()

plt.figure(100, figsize=(18, 5))
cases_seas.plot()
plt.title("seasonal component")
#plt.show()


cases_train = cases[:-24]
cases_test = cases[-24:]

# scale the time series on the training settransformer = Scaler()
transformer = Scaler()
cases_ttrain = transformer.fit_transform(cases_train)
cases_ttest = transformer.transform(cases_test)
cases_t = transformer.transform(cases)


#create covariates

#cov = TimeSeries.from_dataframe(state_data)

# create covariates: year, month, and integer index series
cov = datetime_attribute_timeseries(cases, attribute="year", one_hot=False)
cov = cov.stack(datetime_attribute_timeseries(cases, attribute="month", one_hot=False))
cov = cov.stack(TimeSeries.from_times_and_values(
                                    times=cases.time_index,
                                    values=np.arange(len(cases)),
                                    columns=["linear_increase"]))
cov = cov.astype(np.float32)


train_cov = cov[:-24]
test_cov = cov[-24:]

# rescale the covariates: fit on the training set
scaler = Scaler()
scaler.fit(train_cov)
tcov = scaler.transform(cov)



# naive seasonal forecast
modelNs = NaiveSeasonal(K=mseas)
modelNs.fit(cases_train)
ts_predNs = modelNs.predict(len(cases_test))

# naive drift (trend) forecast
modelNd = NaiveDrift()
modelNd.fit(cases_train)
ts_predNd = modelNd.predict(len(cases_test))

ts_predN = ts_predNd + ts_predNs - cases_train.last_value()
plt.clf()
plt.figure(100, figsize=(18, 5))
cases.plot(label="actual")
ts_predN.plot(label="naive forecast")
plt.title("Naive Forecast (MAE: {:.2f}%)".format(mae(cases_test, ts_predN)))
#plt.axvline(x = len(cases_test), linestyle='--', color='red')
plt.legend()
#plt.show()

plt.clf()


# search space for best theta value: check 100 alternatives
modelX = ExponentialSmoothing(
                seasonal_periods=mseas,
                seasonal=ModelMode.ADDITIVE)
modelX.fit(cases_train)
ts_predX = modelX.predict(  n=len(cases_test),
                            num_samples=N_SAMPLES)

plt.figure(100, figsize=(18, 5))
cases.plot(label="actual")
ts_predX.plot(label="Exponential Smoothing")
plt.title("Exponential Smoothing (MAE: {:.2f}%)".format(mae(cases_test, ts_predX)))
plt.legend()
#plt.show()
plt.clf()


#TFT model

model = TFTModel(input_chunk_length=INLEN,
                 output_chunk_length=N_FC,
                 hidden_size=HIDDEN,
                 lstm_layers=LSTMLAYERS,
                 num_attention_heads=ATTHEADS,
                 dropout=DROPOUT,
                 batch_size=BATCH,
                 n_epochs=EPOCHS,
                 likelihood=QuantileRegression(quantiles=QUANTILES),
                 # loss_fn=MSELoss(),
                 random_state=RAND,
                 force_reset=True)

model.fit(cases_ttrain,
          future_covariates=tcov,
          verbose=True)

# testing: generate predictions
cases_tpred = model.predict(n=len(cases_test),
                         num_samples=N_SAMPLES,
                         n_jobs=N_JOBS)

print("Do the predictions constitute a probabilistic time series?", cases_tpred.is_probabilistic)


# testing: helper function: plot predictions
def plot_predict(ts_actual, ts_test, ts_pred):
    ## plot time series, limited to forecast horizon
    plt.figure(figsize=FIGSIZE)

    ts_actual.plot(label="actual")  # plot actual

    ts_pred.plot(low_quantile=qL1, high_quantile=qU1, label=label_q1)  # plot U1 quantile band
    # ts_pred.plot(low_quantile=qL2, high_quantile=qU2, label=label_q2)   # plot U2 quantile band
    ts_pred.plot(low_quantile=qL3, high_quantile=qU3, label=label_q3)  # plot U3 quantile band
    ts_pred.plot(central_quantile="mean", label="expected")  # plot "mean" or median=0.5

    plt.title("TFT: test set (MAE: {:.2f}%)".format(mae(ts_test, ts_pred)))
    plt.legend();

    # testing: call helper function: plot predictions

plt.clf()
cases_pred = transformer.inverse_transform(cases_tpred)
plot_predict(cases, cases_test, cases_pred)
#plt.show()
plt.clf()


# testing: call helper function: plot predictions, focus on test set
cases_pred = transformer.inverse_transform(cases_tpred)
ts_actual = cases[ cases_tpred.start_time(): cases_tpred.end_time() ]  # actual values in forecast horizon
plot_predict(ts_actual, cases_test, cases_pred)
#plt.show()
plt.clf()



# create future covariates: year, month, integer index
FC_HORIZ = 32                       # months after end of training set
start = cases.start_time()
n_per = len(cases_train) + FC_HORIZ    # set a maximum horizon to create covariates for

# create covariates from beginning of training data to end of forecast horizon
cases_year = datetime_attribute_timeseries(
                pd.date_range(start=start, periods=n_per, freq="MS"),  #, closed="right"),
                attribute="year",
                one_hot=False)
cases_month = datetime_attribute_timeseries(
                pd.date_range(start=start, periods=n_per, freq="MS"),
                attribute="month",
                one_hot=False)

cov = cases_year.stack(cases_month)
cov = cov.stack(TimeSeries.from_times_and_values(
                times=cov.time_index,
                values=np.arange(n_per),
                columns=['linear_increase']))

cov = cov.astype(np.float32)
scaler2 = Scaler()
tcov = scaler2.fit_transform(cov)

print("start:", cov.start_time(), "; end:",cov.end_time())

# generate future, out-of-sample predictions
ts_tpred = model.predict(n=FC_HORIZ, future_covariates=tcov, num_samples=N_SAMPLES)
print("start:", ts_tpred.start_time(), "; end:",ts_tpred.end_time())
ts_pred = transformer.inverse_transform(ts_tpred)
plt.figure(figsize=FIGSIZE)

ts_actual = cases.slice_intersect(other=ts_pred)
ts_actual.plot(label="actual")

ts_pred.plot(low_quantile=qL1, high_quantile=qU1, label=label_q1)    # plot U1 quantile band
ts_pred.plot(low_quantile=qL3, high_quantile=qU3, label=label_q3)    # plot U3 quantile band
ts_pred.plot(central_quantile="mean", label="expected")              # plot "mean" or median=0.5
plt.title(  "TFT: forecast" + \
            " from " + format(ts_tpred.start_time(), "%Y.%m") + \
            " to " + format(ts_tpred.end_time(), "%Y.%m"));
plt.legend()
plt.show()