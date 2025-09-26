### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- BARRA FACTORS ----------------------------------------------- ###
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


barra_factors = {
    "Beta": "SPHB",                # High Beta ETF
    "Book-to-Price": "VLUE",       # Value/Book-to-Price ETF
    "Dividend Yield": "VYM",       # High Dividend Yield ETF
    "Earnings Quality": "QUAL",    # Quality ETF
    "Earnings Variability": "QVAL",# Explicit quality/profitability ETF
    "Earnings Yield": "IWD",       # Value ETF (earnings yield as core metric)
    "Growth": "IWF",               # Large Cap Growth ETF
    "Investment Quality": "VFQY",  # Vanguard Quality ETF
    "Leverage": "SPLV",            # Low volatility, proxies low leverage
    "Liquidity": "SPY",            # S&P 500 ETF (liquidity proxy)
    "Mid Capitalization": "IWR",   # Mid Cap ETF
    "Momentum": "MTUM",            # Momentum Style ETF
    "Profitability": "XMMO",       # S&P SmallCap Momentum ETF (profitability proxy)
    "Residual Volatility": "USMV", # Minimum Volatility ETF
    "Size": "IWM"                  # Small Cap ETF
}

def merge_dfs(array_of_dfs):
    return ft.reduce(lambda left, right: pd.merge(left, right,
                                                  left_index=True,
                                                  right_index=True, how='outer'), array_of_dfs)

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- BARRA FACTORS ----------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###




