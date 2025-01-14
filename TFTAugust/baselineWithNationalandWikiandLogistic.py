import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from logistic_fitted import getLogisticPrediction
#os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

EPOCHS = 300
INLEN = 32
HIDDEN = 64
LSTMLAYERS = 2
ATTHEADS = 1
DROPOUT = 0.1
BATCH = 32


N_FC = 6 #number of forecasts
RAND = 42           # set random state
N_SAMPLES = 100     # number of times a prediction is sampled from a probabilistic model
N_JOBS = -1
FIGSIZE = (9, 6)

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
from darts.models import TFTModel
from darts.metrics import mape, mae

from darts.utils.likelihood_models import QuantileRegression


pd.set_option("display.precision",2)
np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = '{:,.2f}'.format

wnv_data = pd.read_csv('../WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

df_results_mae = pandas.DataFrame()
def getData(state):
    state_data = pd.read_csv('../statesAugustSubmission/'+state+'/NOAA_data.csv')
    state_data.index = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(state_data.year, state_data.month)])
    wnvData = pd.read_csv('../statesAugustSubmission/' + state + '/wnv_data.csv', index_col=[0])
    wnvData.index = pd.to_datetime(wnvData.index)
    state_data['count'] = wnvData['count']
    state_data['month_cos'] = np.cos(state_data.index.month * 2 * np.pi / 12)
    state_data['month_sin'] = np.sin(state_data.index.month * 2 * np.pi / 12)
    state_data['real_month_cos'] = np.cos((state_data.index + np.timedelta64(7, 'M')).month * 2 * np.pi / 12)
    state_data['real_month_sin'] = np.sin((state_data.index + np.timedelta64(7, 'M')).month * 2 * np.pi / 12)

    state_data['6monthsAhead'] = state_data['count'].shift(-6)
    state_data['6monthsAgo/1yearbeforePred'] = state_data['count'].shift(6)
    state_data.drop(['count'], axis=1, inplace=True)

    national_count = pd.read_csv('../WNVData/national_count.csv', index_col=[0]).iloc[:,0]
    national_count.index = pd.to_datetime(national_count.index)
    state_data['yearago_national_count'] = national_count
    state_data['yearago_national_count'] = state_data['yearago_national_count'].shift(6)

    wiki_data = pd.read_csv('../WikipediaDataAugust/wiki_data.csv', index_col=[0])
    wiki_data.index = pd.to_datetime(wiki_data.index)
    state_data = pd.concat([state_data, wiki_data], axis=1)

    logistic_values = getLogisticPrediction(state)

    if logistic_values != 0:
        state_data['logistic_prediction'] = logistic_values[6:]

    return state_data

for state in [i for i in wnv_data['state'].unique() if i not in ['DC']]:
#for state in ['CA']:
    state_data = getData(state)

    mosquitoData = pd.read_csv('../MosquitoDataJuly/MonthlyMosquitoData.csv')
    mosquitoData.set_index(pd.to_datetime([f'{y}-{m}-01' for y, m in zip(mosquitoData.year, mosquitoData.month)]), inplace=True)
    state_data = pd.concat([state_data, mosquitoData], axis=1)

    state_data = state_data.dropna().astype('float32')

    ts = TimeSeries.from_series(state_data['6monthsAhead'])
    state_data.drop(['6monthsAhead'], axis=1, inplace=True)


    testStateData = state_data[-6:]
    ts_train = ts[:-6]
    ts_test = ts[-6:]

    transformer = Scaler()
    ts_ttrain = transformer.fit_transform(ts_train)
    ts_ttest = transformer.transform(ts_test)
    ts_t = transformer.transform(ts)

    #Now we deal with covariates

    cov = TimeSeries.from_dataframe(state_data)
    train_cov = cov[:-6]
    test_cov = cov[-6:]

    scaler = Scaler()
    scaler.fit(train_cov)
    tcov = scaler.transform(cov)

    #now we train the model

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
                     pl_trainer_kwargs={
                         "accelerator": "gpu",
                         "devices": [1],
                         #"precision": '32-true'
                     },
                     force_reset=True)

    model.fit(ts_ttrain,
              future_covariates=tcov,
              verbose=True)

    # testing: generate predictions
    ts_tpred = model.predict(n=len(ts_test),
                             num_samples=N_SAMPLES,
                             n_jobs=N_JOBS)
    ts_pred = transformer.inverse_transform(ts_tpred)
    ts_pred = ts_pred[-6:]

    dfY = pd.DataFrame()


    def plot_predict(ts_actual, ts_test, ts_pred):
        ## plot time series, limited to forecast horizon
        plt.figure(figsize=FIGSIZE)

        ts_actual.plot(label="actual")  # plot actual

        ts_pred.plot(low_quantile=qL1, high_quantile=qU1, label=label_q1)  # plot U1 quantile band
        # ts_pred.plot(low_quantile=qL2, high_quantile=qU2, label=label_q2)   # plot U2 quantile band
        ts_pred.plot(low_quantile=qL3, high_quantile=qU3, label=label_q3)  # plot U3 quantile band
        ts_pred.plot(central_quantile="mean", label="expected")  # plot "mean" or median=0.5

        plt.title("TFT: test set (MAE: {:.2f})".format(mae(ts_test, ts_pred)))
        plt.legend();

    plot_predict(ts, ts_test, ts_pred)
    df_results_mae = df_results_mae.append({'state': state, 'withNationalandWiki': mae(ts_test, ts_pred)}, ignore_index=True)
    #plt.show()
    plt.savefig('../statesAugustSubmission/'+state+'/train+testwithNationalandWikiandLogistic.png')
    plt.clf()
    ts_pred = transformer.inverse_transform(ts_tpred)
    ts_actual = ts[ts_tpred.start_time(): ts_tpred.end_time()]  # actual values in forecast horizon
    plot_predict(ts_actual, ts_test, ts_pred)
    #plt.show()
    plt.savefig('../statesAugustSubmission/'+state+'/testwithNationalandWikiandLogistic.png')
    plt.clf()
        # helper method: calculate percentiles of predictions
    def predQ(ts_tpred, q):
        ts = ts_pred.quantile_timeseries(q)  # percentile of predictions
        s = TimeSeries.pd_series(ts)
        header = format(int(q * 1000))
        dfY[header] = s


    # call helper function: percentiles of predictions
    quantiles = QUANTILES
    _ = [predQ(ts_tpred, q) for q in quantiles]

    dfY.index = dfY.index+pd.DateOffset(months=6)
    dfY.to_csv('../statesAugustSubmission/'+state+'/withArbovirusWithNationalandWikiandLogistic.csv')
    dfY = dfY[-6:]


df_results_mae.to_csv('../modelResults/August/baselineWithNationalwithWikiwithLogisticTest.csv')

