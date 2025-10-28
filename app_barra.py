### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- BARRA FACTORS ----------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
import plotly.subplots as sp
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
import functools as ft
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
DATA_DIR = os.getenv('DATA_DIR', 'data')

barra_factors = {
    "SPHB": "beta",                # High Beta ETF
    "VLUE": "book_to_price",       # Value/Book-to-Price ETF
    "VYM": "dividend_yield",       # High Dividend Yield ETF
    "IWD": "earnings_yield",       # Value ETF (earnings yield as core metric)
    "IWF": "growth",               # Large Cap Growth ETF
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

with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
    sp500 = pd.read_csv(file)
    sp500.index = pd.to_datetime(sp500['Date']).values
    sp500.drop('Date', axis=1, inplace=True)
    spx_daily = pd.DataFrame(sp500['Close'])
    spx_daily.columns = ['spx']

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

    ### PLOT ###
    columns_to_plot = barra_factors_df.columns
    fig = sp.make_subplots(rows=4, cols=3, subplot_titles=columns_to_plot)
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

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- BARRA FACTORS ----------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_barra_predictor():
    target_feature_df = merge_dfs([
        spx_daily.resample('ME').last().pct_change(),
        barra_factors_df.resample('ME').last().pct_change()]).dropna()
    factor_features = target_feature_df.columns[1:]
    window = 12
    barra_factor_backtest = pd.DataFrame()
    for each_factor in factor_features:
        factor_df = target_feature_df[[each_factor,'spx']].dropna()
        factor_df['correlation'] = (
            factor_df[each_factor]
            .rolling(window, min_periods=window)
            .corr(factor_df['spx'])
            .shift(1)  # equivalent to your .shift(1) at the end
        )
        mean = factor_df['correlation'].rolling(window).mean().shift(1)
        std = factor_df['correlation'].rolling(window).std(ddof=0)
        factor_df['correlation_zscore'] = (factor_df['correlation'] - mean) / std

        ### BACKTEST ###
        conditions = [
            (factor_df['correlation_zscore'] >= 3),
            (factor_df['correlation_zscore'] >= 1) & (factor_df['correlation_zscore'] < 2),
            (factor_df['correlation_zscore'] >= 0) & (factor_df['correlation_zscore'] < 1),
            (factor_df['correlation_zscore'] <= 0) & (factor_df['correlation_zscore'] > -1),
            (factor_df['correlation_zscore'] <= -1) & (factor_df['correlation_zscore'] > -2),
            (factor_df['correlation_zscore'] <= -3)
        ]
        choices = [
            -1 * factor_df['spx'],  # 50%
            -0.75 * factor_df['spx'],  # 50%
            -0.5 * factor_df['spx'],  # 100%
            0.5 * factor_df['spx'],  # -50%
            0.75 * factor_df['spx'],  # -100%
            1.0 * factor_df['spx']  # -100%
        ]

        factor_df['bt_returns'] = np.select(conditions, choices, default=np.nan)
        bt_results = pd.DataFrame(factor_df['bt_returns']).dropna()
        bt_results.columns = [each_factor]
        barra_factor_backtest = merge_dfs([barra_factor_backtest,bt_results])














