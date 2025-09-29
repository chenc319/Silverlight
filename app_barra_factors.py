### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- BARRA FACTORS ----------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
import pandas as pd
import functools as ft
import streamlit as st
import plotly.graph_objs as go
from pathlib import Path
import os
from matplotlib.colors import LinearSegmentedColormap
from plotly.subplots import make_subplots
import numpy as np
import plotly.subplots as sp
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

def plot_barra_factors(start, end, **kwargs):
    df = barra_factors_df.copy().resample('ME').last()
    # Define a gentle neutral palette (no neons)
    palette = [
        "#35c9c3",  # Teal
        "#f9c6bb",  # Peach
        "#98e3f9",  # Light Blue
        "#59b758",  # Leaf Green
        "#e54d42",  # Soft Red
        "#fff8a9",  # Pale Yellow
        "#c4b7f4",  # Lavender
        "#bbf6c2",  # Mint Green
        "#ecbe9d",  # Apricot
        "#6bb7f4",  # Sky Blue
    ]

    # List columns to plot (replace with your actual DataFrame column list)
    columns_to_plot = barra_factors_df.columns

    fig = sp.make_subplots(rows=5, cols=3, subplot_titles=columns_to_plot)

    for i, col in enumerate(columns_to_plot):
        row = i // 3 + 1
        col_pos = i % 3 + 1
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=col,
                line=dict(color=palette[i % len(palette)], width=2)
            ),
            row=row,
            col=col_pos
        )

    for row in range(1, 6):
        for col in range(1, 4):
            fig.update_xaxes(title_text="Date", row=row, col=col)
            fig.update_yaxes(title_text="Value", row=row, col=col)

    fig.update_layout(
        showlegend=False,
        height=1800,
        width=1200
    )
    st.plotly_chart(fig, use_container_width=True)
    barra_factors_pct = barra_factors_df.pct_change()


