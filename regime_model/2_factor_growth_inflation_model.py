### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------- REGIME MODEL ---------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
import functools as ft
from pandas_datareader import data as pdr
from pathlib import Path
import os
import pickle
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from polygon import RESTClient
DATA_DIR = os.getenv('DATA_DIR', 'data')

def merge_dfs(array_of_dfs):
    return ft.reduce(lambda left,
                            right: pd.merge(left,
                                            right,
                                            left_index=True,
                                            right_index=True,
                                            how='outer'),
                     array_of_dfs)

def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.shift(1).rolling(window=window, min_periods=window).mean()
    rolling_std = series.shift(1).rolling(window=window, min_periods=window).std()
    zscore = (series - rolling_mean) / rolling_std
    return zscore
### ---------------------------------------------------------------------------------------------------------- ###
### ------------------------------------------------ DATA PULL ----------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### DATA PULL ###
start = '1900-01-01'
end = pd.to_datetime('today')

with open(Path(DATA_DIR) / 'sp500.csv', 'rb') as file:
    sp500 = pd.read_csv(file)
sp500.index = pd.to_datetime(sp500['Date']).values
sp500.drop('Date', axis=1, inplace=True)
sp500.columns = ['close']
sp500 = sp500.resample('ME').last()

with open(Path(DATA_DIR) / 'AGG.csv', 'rb') as file:
    agg = pd.read_csv(file)
agg.index = pd.to_datetime(agg['Date']).values
agg = pd.DataFrame(agg['Close']).resample('ME').last()

### PREPARE FEATURES ###
growth = pdr.DataReader('USALOLITOAASTSAM', 'fred', start, end).resample('ME').last()
inflation = pdr.DataReader('CPIAUCSL', 'fred', start, end).resample('ME').last()
plt.plot(growth)
plt.show()

growth_inflation_df = merge_dfs([growth,inflation,sp500,agg]).dropna()
growth_inflation_df.columns = ['growth','inflation','sp500','bonds']
growth_inflation_df['growth_roc'] = growth_inflation_df['growth'].diff(1)
growth_inflation_df['growth_z'] = rolling_zscore(growth_inflation_df['growth'],12)
growth_inflation_df['inflation_roc'] = growth_inflation_df['inflation'].diff(1)
growth_inflation_df['inflation_z'] = rolling_zscore(growth_inflation_df['inflation_roc'],12)
growth_inflation_df['sp500_pct'] = growth_inflation_df['sp500'].pct_change()
growth_inflation_df['bonds_pct'] = growth_inflation_df['bonds'].pct_change()
growth_inflation_df = growth_inflation_df.dropna()

plt.plot(growth_inflation_df['growth'])
plt.show()

### ---------------------------------------------------------------------------------------------------------- ###
### ------------------------------------------------ ANALYSIS ------------------------------------------------ ###
### ---------------------------------------------------------------------------------------------------------- ###

reflation_regime = growth_inflation_df[
    (growth_inflation_df['inflation_roc'] > 0) &
    (growth_inflation_df['growth_roc'] > 0)
]
inflation_regime = growth_inflation_df[
    (growth_inflation_df['inflation_roc'] > 0) &
    (growth_inflation_df['growth_roc'] < 0)
]
goldilocks_regime = growth_inflation_df[
    (growth_inflation_df['inflation_roc'] < 0) &
    (growth_inflation_df['growth_roc'] > 0)
]
deflation_regime = growth_inflation_df[
    (growth_inflation_df['inflation_roc'] < 0) &
    (growth_inflation_df['growth_roc'] < 0)
]

reflation_averages = reflation_regime[['sp500_pct','bonds_pct']].mean(axis=0) * 100
inflation_averages = inflation_regime[['sp500_pct','bonds_pct']].mean(axis=0)*  100
goldilocks_averages = goldilocks_regime[['sp500_pct','bonds_pct']].mean(axis=0) * 100
deflation_averages = deflation_regime[['sp500_pct','bonds_pct']].mean(axis=0) * 100

gi_2_factor_results = pd.DataFrame()
gi_2_factor_results['equities'] = [goldilocks_averages[0],
                                   reflation_averages[0],
                                   inflation_averages[0],
                                   deflation_averages[0]]
gi_2_factor_results['bonds'] = [goldilocks_averages[1],
                                reflation_averages[1],
                                inflation_averages[1],
                                deflation_averages[1]]


plt.plot(growth_inflation_df['growth_roc'])
plt.show()





