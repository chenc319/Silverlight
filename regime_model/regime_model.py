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

### ---------------------------------------------------------------------------------------------------------- ###
### ------------------------------------------------ DATA PULL ----------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### DATA PULL ###
start = '1900-01-01'
end = pd.to_datetime('today')

gdp = pdr.DataReader('GDP', 'fred', start, end).resample('ME').last().ffill()
with open(Path(DATA_DIR) / 'gdp.pkl', 'wb') as file:
    pickle.dump(gdp, file)

us_composite_leading_index = pdr.DataReader('USALOLITONOSTSAM', 'fred', start, end).resample('ME').last()
with open(Path(DATA_DIR) / 'us_composite_leading_index.pkl', 'wb') as file:
    pickle.dump(us_composite_leading_index, file)

us_leading_index = pdr.DataReader('USSLIND', 'fred', start, end).resample('ME').last()
with open(Path(DATA_DIR) / 'us_leading_index.pkl', 'wb') as file:
    pickle.dump(us_leading_index, file)

cpi = pdr.DataReader('CPIAUCSL', 'fred', start, end).resample('ME').last()
with open(Path(DATA_DIR) / 'cpi.pkl', 'wb') as file:
    pickle.dump(cpi, file)

unemp = pdr.DataReader('UNRATE', 'fred', start, end).resample('ME').last()
with open(Path(DATA_DIR) / 'unemp.pkl', 'wb') as file:
    pickle.dump(unemp, file)

yield10 = pdr.DataReader('GS10', 'fred', start, end).resample('ME').last()
with open(Path(DATA_DIR) / 'yield10.pkl', 'wb') as file:
    pickle.dump(yield10, file)

yield2 = pdr.DataReader('GS2', 'fred', start, end).resample('ME').last()
with open(Path(DATA_DIR) / 'yield2.pkl', 'wb') as file:
    pickle.dump(yield2, file)

with open(Path(DATA_DIR) / 'sp500.csv', 'rb') as file:
    sp500 = pd.read_csv(file)
sp500.index = pd.to_datetime(sp500['Date']).values
sp500.drop('Date', axis=1, inplace=True)
sp500.columns = ['close']

aggregate_bond_index_proxy = pdr.DataReader('BAMLCC0A0AAAUS', 'fred', start, end).resample('ME').last()
with open(Path(DATA_DIR) / 'gdp.pkl', 'wb') as file:
    pickle.dump(gdp, file)

### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------------- ETA AND EDA ---------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

yield_merge = merge_dfs([yield10, yield2])
yield_merge['diff'] = yield_merge['GS10'] - yield_merge['GS2']
# Feature engineering
sp500["returns"] = sp500["close"].pct_change().rolling(window=21).mean()
sp500["volatility"] = sp500["close"].pct_change().rolling(window=21).std()
features = merge_dfs([
    sp500[["returns", "volatility"]],
    gdp.pct_change().reindex(sp500.index, method="ffill"),
    cpi.pct_change().reindex(sp500.index, method="ffill"),
    unemp.reindex(sp500.index, method="ffill"),
    pd.DataFrame(yield_merge['diff']).reindex(sp500.index, method="ffill")]).dropna()

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(features)

# Fit HMM for regime detection
hmm = GaussianHMM(n_components=4, covariance_type="full", n_iter=1000, random_state=42)
hmm.fit(X)
hidden_states = hmm.predict(X)


# Add regimes
features["regime"] = hidden_states

# Visualize regimes
plt.figure(figsize=(15,6))
for i in range(4):
    plt.plot(features.index[features["regime"] == i], features["returns"][features["regime"] == i], ".", label=f"Regime {i}")
plt.legend()
plt.title("Equity Regimes Detected (S&P 500 Returns + Macros)")
plt.show()


