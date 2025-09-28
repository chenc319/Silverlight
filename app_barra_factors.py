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
    "SPHB": "beta",                # High Beta ETF
    "VLUE": "book_to_price",       # Value/Book-to-Price ETF
    "VYM": "dividend_yield",       # High Dividend Yield ETF
    "QUAL": "earnings_quality",    # Quality ETF
    "QVAL": "earnings_variability",# Explicit quality/profitability ETF
    "IWD": "earnings_yield",       # Value ETF (earnings yield as core metric)
    "IWF": "growth",               # Large Cap Growth ETF
    "VFQY": "investment_quality",  # Vanguard Quality ETF
    "SPLV": "leverage",            # Low volatility, proxies low leverage
    "SPY": "liquidity",            # S&P 500 ETF (liquidity proxy)
    "IWR": "mid_cap",              # Mid Cap ETF
    "MTUM": "momentum",            # Momentum Style ETF
    "XMMO": "profitability",       # S&P SmallCap Momentum ETF (profitability proxy)
    "USMV": "residual_volatility", # Minimum Volatility ETF
    "IWM": "size"                  # Small Cap ETF
}

def merge_dfs(array_of_dfs):
    return ft.reduce(lambda left, right: pd.merge(left, right,
                                                  left_index=True,
                                                  right_index=True, how='outer'), array_of_dfs)

barra_factors_df = pd.DataFrame()
for each_factor in list(barra_factors.keys()):
    with open(Path(DATA_DIR) / (each_factor + '.csv'), 'rb') as file:
        factor_df = pd.read_csv(file)
    factor_df.index = pd.to_datetime(factor_df['Date']).values
    factor_df = pd.DataFrame(factor_df['Close'])
    factor_df.columns = [barra_factors[each_factor]]
    barra_factors_df = merge_dfs([barra_factors_df, factor_df])

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- BARRA FACTORS ----------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_factors(start, end, **kwargs):
    barra_factors_pct = barra_factors_df.pct_change()


