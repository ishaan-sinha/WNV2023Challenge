from darts.utils.statistics import extract_trend_and_seasonality
from darts.utils.utils import ModelMode

EPOCHS = 300
INLEN = 32
HIDDEN = 64
LSTMLAYERS = 2
ATTHEADS = 1
DROPOUT = 0.1
BATCH = 32


N_FC = 12 #number of forecasts
RAND = 42           # set random state
N_SAMPLES = 100     # number of times a prediction is sampled from a probabilistic model
N_JOBS = -1
FIGSIZE = (9, 6)
#MSEAS = 60
mseas = 12

QUANTILES = [0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600,
             0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990]

qL1, qL2, qL3 = 0.01, 0.05, 0.10        # percentiles of predictions: lower bounds
qU1, qU2, qU3 = 1-qL1, 1-qL2, 1-qL3     # upper bounds derived from lower bounds
label_q1 = f'{int(qU1 * 100)} / {int(qL1 * 100)} percentile band'
label_q2 = f'{int(qU2 * 100)} / {int(qL2 * 100)} percentile band'
label_q3 = f'{int(qU3 * 100)} / {int(qL3 * 100)} percentile band'


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel, NaiveSeasonal, NaiveDrift
from darts.metrics import mape, mae

from darts.utils.likelihood_models import QuantileRegression

wnv_data = pd.read_csv('../WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

def getData(state):
    state_data = pd.read_csv('../statesNormal/'+state+'/NOAA_data.csv')
    state_data.index = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(state_data.year, state_data.month)])
    temporalData = pd.read_csv('../statesNormal/' + state + '/temporalData.csv', index_col=[0])
    temporalData.index = pd.to_datetime(temporalData.index)
    state_data['count'] = temporalData['count']
    state_data['month_cos'] = np.cos(state_data.index.month * 2 * np.pi / 12)
    state_data['month_sin'] = np.sin(state_data.index.month * 2 * np.pi / 12)

    state_data['9monthsAhead'] = state_data['count'].shift(-9)
    #state_data['3monthsAgo/1yearbeforePred'] = state_data['count'].shift(3)
    state_data.drop(['count'], axis=1, inplace=True)

    return state_data


for state in ['CA']:
    state_data = getData(state)
    state_data = state_data.dropna()
    # We will make 12 forecasts, as we have 9 months ahead for the rest of the data
    ts = TimeSeries.from_series(state_data['9monthsAhead'])
    state_data.drop(['9monthsAhead'], axis=1, inplace=True)
    testStateData = state_data[-8:]
    ts_train = ts[:-12]
    ts_test = ts[-12:]

    transformer = Scaler()
    ts_ttrain = transformer.fit_transform(ts_train)
    ts_ttest = transformer.transform(ts_test)
    ts_t = transformer.transform(ts)

    # ETS analysis: try to discern trend and seasonal components
    ts_trend, ts_seas = extract_trend_and_seasonality(ts=ts, freq=mseas, model = ModelMode.ADDITIVE)

    plt.figure(100, figsize=(18, 5))
    ts_trend.plot()
    plt.title("trend component")
    plt.show()
    plt.clf()
    plt.figure(100, figsize=(18, 5))
    ts_seas.plot()
    plt.title("seasonal component")
    plt.show()
    plt.clf()


    modelNs = NaiveSeasonal(K=mseas)
    modelNs.fit(ts_train)
    ts_predNs = modelNs.predict(len(ts_test))

    # naive drift (trend) forecast
    modelNd = NaiveDrift()
    modelNd.fit(ts_train)
    ts_predNd = modelNd.predict(len(ts_test))

    ts_predN = ts_predNd + ts_predNs - ts_train.last_value()

    plt.figure(100, figsize=(18, 5))
    ts.plot(label="actual")
    ts_predN.plot(label="naive forecast")
    plt.title("Naive Forecast (MAE: {:.2f})".format(mae(ts_test, ts_predN)))
    plt.legend()
    plt.show()