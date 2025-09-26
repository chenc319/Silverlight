### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME FACTORS ---------------------------------------------- ###
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

quad_regime_factors = {
    "SPHB": "high_beta",
    "SPLV": "low_beta",
    "IWM": "small_caps",
    "IWR": "mid_caps",
    "MGK": "mega_cap_growth",
    "IYT": "cyclicals", # or IWN
    "DEF": "defensives",
    "OEF": "size",
    "QUAL": "quality",
    "SPHD": "dividends",
    "MTUM": "momentum",
    "IWD": "value",
    "IWF": "growth",
    "IWB": "large_caps"
}
def merge_dfs(array_of_dfs):
    return ft.reduce(lambda left, right: pd.merge(left, right,
                                                  left_index=True,
                                                  right_index=True, how='outer'), array_of_dfs)

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME FACTORS ---------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_quad_regime_factors(start,end,**kwargs):
    all_quad_regime_factors = pd.DataFrame()
    for each_factor in list(quad_regime_factors.keys()):
        with open(Path(DATA_DIR) / (each_factor + '.csv'), 'rb') as file:
            df = pd.read_csv(file)
        df.index = pd.to_datetime(df['Date']).values
        final_df = pd.DataFrame(df['Close'])
        final_df.columns = [quad_regime_factors[each_factor]]
        all_quad_regime_factors = merge_dfs([all_quad_regime_factors,final_df])





