### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------- FED PLUMBING ---------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
import functools as ft
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
DATA_DIR = os.getenv('DATA_DIR', 'data')

def merge_dfs(array_of_dfs):
    return ft.reduce(lambda left, right: pd.merge(left, right,
                                                  left_index=True,
                                                  right_index=True, how='outer'), array_of_dfs)

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------- FED PLUMBING ---------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_liquidity_signal():
    with open(Path(DATA_DIR) / 'di_reserves.pkl', 'rb') as file:
        di_reserves = pd.read_pickle(file)
    with open(Path(DATA_DIR) / 'm2_money_supply.pkl', 'rb') as file:
        m2_money_supply = pd.read_pickle(file)
    with open(Path(DATA_DIR) / 'currency_in_circulation.pkl', 'rb') as file:
        currency_in_circulation = pd.read_pickle(file)
    with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
        sp500 = pd.read_csv(file)
    sp500.index = pd.to_datetime(sp500['Date']).values
    sp500.drop('Date', axis=1, inplace=True)
    spx_monthly = pd.DataFrame(sp500['Close']).resample('ME').last()
    spx_monthly.columns = ['spx']
    spx_monthly_pct = spx_monthly.pct_change().dropna()

    spx_fed_plumbing_merge = merge_dfs([
        spx_monthly_pct.shift(-1),
        di_reserves,
        m2_money_supply,
        currency_in_circulation
    ])
    spx_fed_plumbing_merge[spx_fed_plumbing_merge.columns[1:]] = spx_fed_plumbing_merge[spx_fed_plumbing_merge.columns[1:]].diff()
    spx_fed_plumbing_merge = spx_fed_plumbing_merge.dropna()

    factor_features = spx_fed_plumbing_merge.columns[1:]
    window = 63
    spx_fed_plumbing_merge_mean = spx_fed_plumbing_merge.rolling(window=window).mean()
    spx_fed_plumbing_merge_std = spx_fed_plumbing_merge.rolling(window=window).std()
    spx_fed_plumbing_merge_z = (spx_fed_plumbing_merge - spx_fed_plumbing_merge_mean) / spx_fed_plumbing_merge_std
    spx_fed_plumbing_merge_z = spx_fed_plumbing_merge_z['2005-01-01':]

    bucket_edges = [-np.inf, -3, -2, -1, 0, 1, 2, 3, np.inf]
    bucket_labels = ['<-3', '-3 to -2', '-2 to -1', '-1 to 0', '0 to 1', '1 to 2', '2 to 3', '>3']

    results = {}

    for col in ['TOTRESNS', 'M2SL', 'CURRCIR']:
        # Assign buckets based on z-scores
        bucketed = pd.cut(spx_fed_plumbing_merge_z[col], bins=bucket_edges, labels=bucket_labels, include_lowest=True)
        # Group by bucket and compute average SPX
        avg_spx_by_bucket = spx_fed_plumbing_merge_z.groupby(bucketed)['spx'].mean()
        results[col] = avg_spx_by_bucket

    # Display the tables for each factor
    for factor, result in results.items():
        print(f"Average SPX return by {factor} z-score bucket:")
        print(result)
        print()

    # Optionally, convert results to a single DataFrame for easier viewing
    avg_return_df = pd.DataFrame(results)
    print(avg_return_df)







