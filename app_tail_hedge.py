### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------- TAIL HEDGE PORTFOLIO ------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
import pandas as pd
import functools as ft
import streamlit as st
import plotly.graph_objs as go
from pandas_datareader import data as pdr
from matplotlib import pyplot as plt
from pathlib import Path
import os
import pickle
from plotly.subplots import make_subplots
import numpy as np
DATA_DIR = os.getenv('DATA_DIR', 'data')

def merge_dfs(array_of_dfs):
    return ft.reduce(lambda left, right: pd.merge(left, right,
                                                  left_index=True,
                                                  right_index=True, how='outer'), array_of_dfs)

### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------- TAIL HEDGE PORTFOLIO ------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_veqtor_vix(start, end, **kwargs):
    def compute_ivt(df, window=10):
        d_ivt = df['divt'].astype(int).values
        ivt_trend = np.zeros_like(d_ivt)
        for i in range(window - 1, len(d_ivt)):
            window_slice = d_ivt[i - window + 1:i + 1]
            if np.all(window_slice == 1):
                ivt_trend[i] = 1
            elif np.all(window_slice == -1):
                ivt_trend[i] = -1
            else:
                ivt_trend[i] = 0
        # Add to DataFrame
        df['IVT'] = ivt_trend
        return df
    def assign_weight(rv, ivt):
        if rv < 0.10:
            if ivt == -1:
                return 0.025
            elif ivt == 0:
                return 0.025
            elif ivt == 1:
                return 0.10
        elif 0.10 <= rv < 0.20:
            if ivt == -1:
                return 0.025
            elif ivt == 0:
                return 0.10
            elif ivt == 1:
                return 0.15
        elif 0.20 <= rv < 0.35:
            if ivt == -1:
                return 0.10
            elif ivt == 0:
                return 0.15
            elif ivt == 1:
                return 0.25
        elif 0.35 <= rv <= 0.45:
            if ivt == -1:
                return 0.15
            elif ivt == 0:
                return 0.25
            elif ivt == 1:
                return 0.40
        elif rv > 0.45:
            if ivt == -1:
                return 0.25
            elif ivt == 0:
                return 0.40
            elif ivt == 1:
                return 0.40
        return np.nan  # Default if none match
    with open(Path(DATA_DIR) / 'VIX.csv', 'rb') as file:
        vix = pd.read_csv(file)
    with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
        spx = pd.read_csv(file)

    ### CALCULATE DATA ###
    vix.index = pd.to_datetime(vix['Date']).values
    vix_df = pd.DataFrame(vix['Close'])
    vix_df.columns = ['vix']
    vix_df = vix_df.dropna()
    spx.index = pd.to_datetime(spx['Date']).values
    spx = pd.DataFrame(spx['Close'])
    spx.columns = ['spx']
    spx = spx.dropna()

    rv_df = pd.DataFrame(
        spx.pct_change().rolling(window=21).std() * 252**0.5
    )
    rv_df.columns = ['rv']
    vix_df['5d'] = vix_df['vix'].rolling(window=5).mean()
    vix_df['20d'] = vix_df['vix'].rolling(window=20).mean()
    vix_df['divt'] = np.nan
    for row in vix_df.index:
        if vix_df.loc[row,'5d'] >= vix_df.loc[row,'20d']:
            vix_df.loc[row, 'divt'] = 1
        elif vix_df.loc[row,'5d'] < vix_df.loc[row,'20d']:
            vix_df.loc[row, 'divt'] = -1
    vix_df = compute_ivt(vix_df.dropna())
    vix_df = merge_dfs([vix_df,rv_df])
    vix_df['weights'] = vix_df.apply(
        lambda row: assign_weight(row['rv'], row['IVT']), axis=1)
    vix_df = vix_df.dropna()

    ### PLOT ###
    fig = make_subplots(rows=1, cols=2, subplot_titles=["RV", "IV"])
    fig.add_trace(
        go.Scatter(
            x=vix_df.index, y=vix_df["rv"], mode="lines",
            line=dict(color="#2874A6", width=2), name="RV"
        ),
        row=1, col=1
    )
    fig.update_yaxes(title_text="Annualized 1m SPX Volatility", row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=vix_df.index, y=vix_df["vix"], mode="lines",
            line=dict(color="#E67E22", width=2), name="IV"
        ),
        row=1, col=2
    )
    fig.update_yaxes(title_text="VIX", row=1, col=2)
    fig.update_layout(height=400, width=900, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    ### PLOT ###
    fig = make_subplots(rows=1, cols=2, subplot_titles=["5d IV", "20d IV"])
    fig.add_trace(
        go.Scatter(
            x=vix_df.index, y=vix_df["5d"], mode="lines",
            line=dict(color="#2874A6", width=2), name="5d IV"
        ),
        row=1, col=1
    )
    fig.update_yaxes(title_text="Rolling 5d VIX", row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=vix_df.index, y=vix_df["20d"], mode="lines",
            line=dict(color="#E67E22", width=2), name="20d IV"
        ),
        row=1, col=2
    )
    fig.update_yaxes(title_text="Rolling 20d VIX", row=1, col=2)
    fig.update_layout(height=400, width=900, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    ### PLOT ###
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vix_df.index,
        y=vix_df['weights'] * 100,
        mode='lines',
        line=dict(color='#29B6D9', width=2)))
    fig.update_layout(
        title='RV/IV Cross Weights'
    )
    st.plotly_chart(fig, use_container_width=True)

















