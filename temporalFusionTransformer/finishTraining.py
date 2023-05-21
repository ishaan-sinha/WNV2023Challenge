import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

EPOCHS = 3
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


for state in [i for i in wnv_data['state'].unique() if i not in ['DC']]:
    state_data = pd.read_csv('../statesMaySubmission/'+state+'/NOAA_data.csv')
    state_data.index = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(state_data.year, state_data.month)])
    temporalData = pd.read_csv('../statesMaySubmission/'+state+'/temporalData.csv', index_col=[0])
    temporalData.index = pd.to_datetime(temporalData.index)
    state_data['count'] = temporalData['count']

    state_data['month_cos'] = np.cos(state_data.index.month * 2 * np.pi / 12)
    state_data['month_sin'] = np.sin(state_data.index.month * 2 * np.pi / 12)

    state_data['8monthsAhead'] = state_data['count'].shift(-8)
    state_data.drop(['count'], axis=1, inplace=True)

    train_data = state_data[:-11]
    test_data = state_data[-11:]
    print(len(test_data))


    toPredict = train_data['8monthsAhead']

    train_data.drop(['8monthsAhead'], axis=1, inplace=True)
    test_data.drop(['8monthsAhead'], axis=1, inplace=True)

    cases = TimeSeries.from_series(toPredict)

    cases_train = cases

    # scale the time series on the training settransformer = Scaler()
    transformer = Scaler()
    cases_ttrain = transformer.fit_transform(cases_train)


    #create covariates

    train_cov = TimeSeries.from_dataframe(train_data)

    test_cov = TimeSeries.from_dataframe(test_data)

    # rescale the covariates: fit on the training set
    scaler = Scaler()
    scaler.fit(train_cov)
    ttrain_cov = scaler.transform(train_cov)
    ttest_cov = scaler.transform(test_cov)

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
                      #loss_fn= MSELoss(),
                     random_state=RAND,
                     force_reset=True)

    model.fit(cases_ttrain,
              future_covariates=ttrain_cov,
              verbose=True)



    # testing: predictions
    cases_tpred = model.predict(n=len(ttest_cov), future_covariates=ttest_cov, num_samples=N_SAMPLES, n_jobs=N_JOBS)
    cases_tpred = transformer.inverse_transform(cases_tpred)

    plt.clf()

    print(cases_tpred)

    import os

    pickle.dump(model, open('../statesMaySubmission/' + state + '/8monthsAhead-OnlyWeatherData-Trained including August 2021-300epochs+train.sav', 'wb'))
    print(state)
    break

