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
    "SPHB": "High Beta",
    "SPLV": "Low Beta",
    "IWM": "Small Caps",
    "IWR": "Mid Caps",
    "MGK": "Mega Cap Growth",
    "IYT": "Cyclicals",
    "IWN": "Cyclicals",
    "DEF": "Defensives",
    "OEF": "Size",
    "QUAL": "Quality",
    "SPHD": "Dividends",
    "MTUM": "Momentum",
    "IWD": "Value",
    "IWF": "Growth",
    "IWB": "Large Caps"
}
def merge_dfs(array_of_dfs):
    return ft.reduce(lambda left, right: pd.merge(left, right,
                                                  left_index=True,
                                                  right_index=True, how='outer'), array_of_dfs)

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME FACTORS ---------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###







