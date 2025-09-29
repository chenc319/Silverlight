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

def plot_vix_vvix(start, end, **kwargs):
    with open(Path(DATA_DIR) / 'VIX.csv', 'rb') as file:
        vix = pd.read_csv(file)
    with open(Path(DATA_DIR) / 'VVIX.csv', 'rb') as file:
        vvix = pd.read_csv(file)

    vix.index = pd.to_datetime(vix['Date']).values
    vix = pd.DataFrame(vix['Close'])
    vix.columns = ['vix']
    vvix.index = pd.to_datetime(vvix['Date']).values
    vvix = pd.DataFrame(vvix['Close'])
    vvix.columns = ['vvix']

    vix_vvix_merge = merge_dfs([vix,vvix])
    fig = go.Figure()

    # VIX on left y-axis
    fig.add_trace(
        go.Scatter(
            x=vix_vvix_merge.index, y=vix_vvix_merge['vix'], name='VIX',
            line=dict(color='#1f77b4', width=2),
            yaxis='y1'
        )
    )

    # VVIX on right y-axis
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['vvix'], name='VVIX',
            line=dict(color='#ff7f0e', width=2),
            yaxis='y2'
        )
    )

    fig.update_layout(
        yaxis=dict(
            title='VIX',
            showgrid=True,
            zeroline=False
        ),
        yaxis2=dict(
            title='VVIX',
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False
        ),
        xaxis=dict(title='Date'),
        legend=dict(x=0.01, y=0.97),
        height=400,
        width=900
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_veqtor_vix(start, end, **kwargs):
    with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
        spx = pd.read_csv(file)
    spx.index = pd.to_datetime(spx['Date']).values
    factor_df = pd.DataFrame(spx['Close'])
    factor_df.columns = ['spx']








