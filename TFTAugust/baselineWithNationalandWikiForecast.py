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


N_FC = 12 #number of forecasts
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

df_results_mae = pandas.DataFrame(columns=['state', 'withArbovirus'])
def getData(state):
    state_data = pd.read_csv('../statesAugustSubmission/'+state+'/NOAA_data.csv')
    state_data.index = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(state_data.year, state_data.month)])
    wnvData = pd.read_csv('../statesAugustSubmission/' + state + '/wnv_data.csv', index_col=[0])
    wnvData.index = pd.to_datetime(wnvData.index)
    state_data['count'] = wnvData['count']
    state_data['month_cos'] = np.cos(state_data.index.month * 2 * np.pi / 12)
    state_data['month_sin'] = np.sin(state_data.index.month * 2 * np.pi / 12)
    state_data['real_month_cos'] = np.cos((state_data.index + np.timedelta64(6, 'M')).month * 2 * np.pi / 12)
    state_data['real_month_sin'] = np.sin((state_data.index + np.timedelta64(6, 'M')).month * 2 * np.pi / 12)

    state_data['6monthsAhead'] = state_data['count'].shift(-6)
    state_data['6monthsAgo/1yearbeforePred'] = state_data['count'].shift(6)
    state_data.drop(['count'], axis=1, inplace=True)

    national_count = pd.read_csv('../WNVData/national_count.csv', index_col=[0]).iloc[:, 0]
    national_count.index = pd.to_datetime(national_count.index)
    state_data['yearago_national_count'] = national_count
    state_data['yearago_national_count'] = state_data['yearago_national_count'].shift(6)

    wiki_data = pd.read_csv('../WikipediaData/wiki_data.csv', index_col=[0])
    wiki_data.index = pd.to_datetime(wiki_data.index)
    state_data = pd.concat([state_data, wiki_data], axis=1)

    return state_data


for state in [i for i in wnv_data['state'].unique() if i not in ['DC']]:
#for state in ['CA']:
    state_data = getData(state)
    mosquitoData = pd.read_csv('../MosquitoDataJuly/MonthlyMosquitoData.csv')
    mosquitoData.set_index(pd.to_datetime([f'{y}-{m}-01' for y, m in zip(mosquitoData.year, mosquitoData.month)]), inplace=True)
    state_data = pd.concat([state_data, mosquitoData], axis=1)
    state_data = state_data[state_data['rate/trap_night.2'].first_valid_index():].astype(np.float32)

    # We will make 12 forecasts, as we have 7 months ahead only for the last 12 months
    ts = TimeSeries.from_series(state_data['6monthsAhead'].dropna())

    state_data.drop(['6monthsAhead', 'year', 'month'], axis=1, inplace=True)
    state_data = state_data.dropna()

    transformer = Scaler()
    ts_ttrain = transformer.fit_transform(ts)

    # Now we deal with covariates

    cov = TimeSeries.from_dataframe(state_data)

    scaler = Scaler()
    scaler.fit(cov)
    tcov = scaler.transform(cov)

    ts_ttrain = ts_ttrain.astype(np.float32)


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
                     force_reset=True,
                     pl_trainer_kwargs={
                         "accelerator": "gpu",
                         "devices": [1],
                          #"precision": '32-true'
                     }
                    )


    model.fit(ts_ttrain,
              future_covariates=tcov,
              verbose=True)

    # testing: generate predictions
    ts_tpred = model.predict(n=12,
                             num_samples=N_SAMPLES,
                             n_jobs=N_JOBS)
    ts_pred = transformer.inverse_transform(ts_tpred)

    ts_pred = ts_pred[-6:]

    dfY = pd.DataFrame()

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
    dfY = dfY[-6:]

    dfY.to_csv('../statesAugustSubmission/' + state + '/FORECASTbaselineWithNationalWithWiki.csv')

    print(state)




