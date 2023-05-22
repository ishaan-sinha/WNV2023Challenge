import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



EPOCHS = 300
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

QUANTILES = [0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600,
             0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990]

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
from darts.models import TFTModel
from darts.metrics import mape, mae

from darts.utils.likelihood_models import QuantileRegression


pd.set_option("display.precision",2)
np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = '{:,.2f}'.format

wnv_data = pd.read_csv('../WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])


#for state in [i for i in wnv_data['state'].unique() if i not in ['DC']]:
for state in ['CA']:
    all_data = pd.read_csv('../statesExtended/'+ state +'/withAllInputs_powerTransformed_' + state +'.csv', index_col=[0])
    cases = all_data['count']
    cases.index = pd.DatetimeIndex(cases.index)

    #state_data = pd.read_csv('../statesExtended/'+state+'/withWeatherInputs_'+state +'.csv', index_col=[0])
    state_data = all_data.drop(['count'], axis=1)
    state_data.index = pd.DatetimeIndex(state_data.index)

    cases = TimeSeries.from_series(cases)

    cases_train = cases[:-24]
    cases_test = cases[-24:]

    # scale the time series on the training settransformer = Scaler()
    transformer = Scaler()
    cases_ttrain = transformer.fit_transform(cases_train)
    cases_ttest = transformer.transform(cases_test)
    cases_t = transformer.transform(cases)


    #create covariates

    cov = TimeSeries.from_dataframe(state_data)

    train_cov = cov[:-24]
    test_cov = cov[-24:]

    # rescale the covariates: fit on the training set
    scaler = Scaler()
    scaler.fit(train_cov)
    tcov = scaler.transform(cov)

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
              future_covariates=tcov,
              verbose=True)



    # testing: predictions
    cases_tpred = model.predict(n=len(cases_test),
                             num_samples=N_SAMPLES,
                             n_jobs=N_JOBS)

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
    plt.show()
    plt.clf()


    # testing: call helper function: plot predictions, focus on test set
    cases_pred = transformer.inverse_transform(cases_tpred)
    ts_actual = cases[ cases_tpred.start_time(): cases_tpred.end_time() ]  # actual values in forecast horizon
    plot_predict(ts_actual, cases_test, cases_pred)
    plt.show()
    plt.clf()
