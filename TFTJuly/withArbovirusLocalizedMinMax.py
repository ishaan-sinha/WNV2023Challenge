import darts.utils.missing_values
import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

#os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

EPOCHS = 300
INLEN = 32
HIDDEN = 64
LSTMLAYERS = 2
ATTHEADS = 1
DROPOUT = 0.1
BATCH = 32


N_FC = 7 #number of forecasts
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
from sklearn.preprocessing import MinMaxScaler
from darts.models import TFTModel
from darts.metrics import mape, mae

from darts.utils.likelihood_models import QuantileRegression
from darts.utils.missing_values import fill_missing_values

pd.set_option("display.precision",2)
np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = '{:,.2f}'.format

wnv_data = pd.read_csv('../WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

df_results_mae = pandas.DataFrame()
def getData(state):
    state_data = pd.read_csv('../statesJulySubmission/'+state+'/NOAA_data.csv')
    state_data.index = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(state_data.year, state_data.month)])
    wnvData = pd.read_csv('../statesJulySubmission/' + state + '/wnv_data.csv', index_col=[0])
    wnvData.index = pd.to_datetime(wnvData.index)
    state_data['count'] = wnvData['count']
    state_data['month_cos'] = np.cos(state_data.index.month * 2 * np.pi / 12)
    state_data['month_sin'] = np.sin(state_data.index.month * 2 * np.pi / 12)
    state_data['real_month_cos'] = np.cos((state_data.index + np.timedelta64(7, 'M')).month * 2 * np.pi / 12)
    state_data['real_month_sin'] = np.sin((state_data.index + np.timedelta64(7, 'M')).month * 2 * np.pi / 12)

    state_data['7monthsAhead'] = state_data['count'].shift(-7)
    state_data['5monthsAgo/1yearbeforePred'] = state_data['count'].shift(5)
    state_data.drop(['count'], axis=1, inplace=True)

    return state_data


for state in [i for i in wnv_data['state'].unique() if i not in ['DC']]:
#for state in ['CA']:
    state_data = getData(state)

    mosquitoData = pd.read_csv('../statesJulySubmission/'+ state +'/MonthlyMosquitoData500miles.csv')
    mosquitoData.set_index(pd.to_datetime([f'{y}-{m}-01' for y, m in zip(mosquitoData.year, mosquitoData.month)]), inplace=True)
    state_data = pd.concat([state_data, mosquitoData], axis=1)
    state_data.drop(['year', 'month'], axis=1, inplace=True)
    state_data = state_data[state_data['7monthsAhead'].notna()]

    #We will make 7 forecasts, as we have 7 months ahead for the rest of the data
    ts = TimeSeries.from_series(state_data['7monthsAhead'])
    state_data.drop(['7monthsAhead'], axis=1, inplace=True)

    state_data = state_data.astype('float32')

    testStateData = state_data[-N_FC:]
    ts_train = ts[:-N_FC]
    ts_test = ts[-N_FC:]

    transformer = Scaler(MinMaxScaler())
    ts_ttrain = transformer.fit_transform(ts_train)
    ts_ttest = transformer.transform(ts_test)
    ts_t = transformer.transform(ts)
    #Now we deal with covariates

    cov = TimeSeries.from_dataframe(state_data)
    train_cov = cov[:-N_FC]
    test_cov = cov[-N_FC:]

    scaler = Scaler(MinMaxScaler())

    scaler.fit(train_cov)
    tcov = scaler.transform(cov)
    train_cov = darts.utils.missing_values.fill_missing_values(train_cov, fill = float(-1))
    tcov = darts.utils.missing_values.fill_missing_values(tcov, fill = float(-1))

    ts_ttrain = ts_ttrain.astype('float32')
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
                         "devices": [0],
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
    ts_pred = ts_pred[-N_FC:]

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
    plt.clf()
    plot_predict(ts, ts_test, ts_pred)
    df_results_mae = df_results_mae.append({'state': state, 'withArbovirusLocalizedMinMax': mae(ts_test, ts_pred)}, ignore_index=True)
    #plt.show()
    plt.savefig('../statesJulySubmission/'+state+'/train+testwithArbovirusLocalizedMinMax.png')
    plt.clf()
    ts_pred = transformer.inverse_transform(ts_tpred)
    ts_actual = ts[ts_tpred.start_time(): ts_tpred.end_time()]  # actual values in forecast horizon
    plot_predict(ts_actual, ts_test, ts_pred)
    #plt.show()
    plt.savefig('../statesJulySubmission/'+state+'/testwithArbovirusLocalizedMinMax.png')
        # helper method: calculate percentiles of predictions
    def predQ(ts_tpred, q):
        ts = ts_pred.quantile_timeseries(q)  # percentile of predictions
        s = TimeSeries.pd_series(ts)
        header = format(int(q * 1000))
        dfY[header] = s


    # call helper function: percentiles of predictions
    quantiles = QUANTILES
    _ = [predQ(ts_tpred, q) for q in quantiles]

    dfY.index = dfY.index+pd.DateOffset(months=N_FC)
    dfY.to_csv('../statesJulySubmission/'+state+'/withArbovirusLocalizedMinMax.csv')
    dfY = dfY[-N_FC:]
    dfY = dfY.reset_index()
    print(state)


df_results_mae.to_csv('../modelResults/July/withArbovirusLocalizedTestMinMax.csv')


