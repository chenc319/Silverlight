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





