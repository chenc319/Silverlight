### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------- FIRV FACTORS ---------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
import pandas as pd
import functools as ft
import streamlit as st
import plotly.graph_objs as go
from pandas_datareader import data as pdr
from pathlib import Path
import os
from matplotlib.colors import LinearSegmentedColormap
from plotly.subplots import make_subplots
import numpy as np
DATA_DIR = os.getenv('DATA_DIR', 'data')

treasury_factors = {
    "treasury_1m": "1m",
    "treasury_3m": "3m",
    "treasury_6m": "6m",
    "treasury_1y": "1y",
    "treasury_2y": "2y",
    "treasury_3y": "3y",
    "treasury_5y": "5y",
    "treasury_7y": "7y",
    "treasury_10y": "10y",
    "treasury_20y": "20y",
    "treasury_30y": "30y"
}
def merge_dfs(array_of_dfs):
    return ft.reduce(lambda left, right: pd.merge(left, right,
                                                  left_index=True,
                                                  right_index=True, how='outer'), array_of_dfs)

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------- FIRV FACTORS ---------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_treasury_yield_curves(start,end,**kwargs):
    treasury_yield_curve = pd.DataFrame()
    for each_factor in list(treasury_factors.keys()):
        with open(Path(DATA_DIR) / (each_factor + '.pkl'), 'rb') as file:
            df = pd.read_pickle(file)
        df.index = pd.to_datetime(df.index).values
        final_df = pd.DataFrame(df['Close'])
        final_df.columns = [quad_regime_factors[each_factor]]
        all_quad_regime_factors = merge_dfs([all_quad_regime_factors, final_df])
