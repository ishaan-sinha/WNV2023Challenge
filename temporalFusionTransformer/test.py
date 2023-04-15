import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wnv_data = pd.read_csv('../states/CA/withWeatherInputs_powerTransformed_CA.csv', index_col=[0])
wnv_data.index = pd.DatetimeIndex(wnv_data.index)
wnv_data.index = pd.DatetimeIndex(wnv_data.index).to_period('M')


EPOCHS = 10
INLEN = 32
HIDDEN = 64
LSTMLAYERS = 2
ATTHEADS = 1
DROPOUT = 0.1
BATCH = 32

N_FC = 36
RAND = 42           # set random state
N_SAMPLES = 100     # number of times a prediction is sampled from a probabilistic model
N_JOBS = -1

QUANTILES = [0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99]

TRAIN = "20200401"  # train/test split
