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
spx_daily = spx_daily.resample('ME').last()

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

def plot_z_score_barra_factors_spx(start,end,**kwarg):
    ### CALCULATE FACTOR RETURNS ###
    barra_factors_pct = barra_factors_df.pct_change().dropna()
    mean_factor_returns = barra_factors_pct.rolling(63).mean()
    std_factor_returns = barra_factors_pct.rolling(63).std()
    barr_factor_z_scores = (barra_factors_pct - mean_factor_returns) / std_factor_returns
    barr_factor_z_spx_merge = merge_dfs([barr_factor_z_scores, spx_daily.pct_change()])

    # replace 'df' with your actual dataframe
    factor_cols = [col for col in barr_factor_z_spx_merge.columns if col != 'spx']
    bin_edges = [-np.inf, -2, -1, 1, 2, np.inf]
    bin_labels = ["<-2", "-2 to -1", "-1 to 1", "1 to 2", ">=2"]

    results = []
    for factor in factor_cols:
        bins = pd.cut(barr_factor_z_spx_merge[factor], bins=bin_edges, labels=bin_labels)
        group = barr_factor_z_spx_merge.groupby(bins)['spx'].mean().reset_index()
        group["factor"] = factor
        group.rename(columns={factor: "z_bin"}, inplace=True)
        results.append(group)

    panel_df = pd.concat(results, ignore_index=True)
    panel_df = panel_df[['factor', panel_df.columns[0], 'spx']]
    panel_df.rename(columns={panel_df.columns[1]: 'z_bin'}, inplace=True)
    panel_df['spx'] = panel_df['spx'].round(6)
    ### PLOT ###
    factors = panel_df['factor'].unique()
    z_bins = ["<-2", "-2 to -1", "-1 to 1", "1 to 2", ">=2"]
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=factors,
        horizontal_spacing=0.08, vertical_spacing=0.07
    )
    for idx, factor in enumerate(factors):
        row = idx // 3 + 1
        col = idx % 3 + 1
        fdata = panel_df[panel_df['factor'] == factor].set_index('z_bin').reindex(z_bins)
        colors = ['red' if v < 0 else 'green' for v in fdata['spx']]
        bar = go.Bar(
            x=z_bins,
            y=fdata['spx'],
            marker_color=colors,
            showlegend=False
        )
        fig.add_trace(bar, row=row, col=col)
        fig.update_yaxes(title_text='Avg SPX Return', row=row, col=col)

    fig.update_layout(
        height=1200, width=1200,
        title_text='Factor-wise Avg SPX Return by Z-Score Bucket',
        bargap=0.15,
    )
    st.plotly_chart(fig)

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- BARRA FACTORS ----------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_kNN_barra_factor_results(start,end,**kwargs):
    from sklearn.neighbors import NearestNeighbors
    ### CALCULATE FACTOR RETURNS ###
    barra_factors_pct = barra_factors_df.pct_change().dropna()
    mean_factor_returns = barra_factors_pct.rolling(252).mean()
    std_factor_returns = barra_factors_pct.rolling(252).std()
    barr_factor_z_scores = (barra_factors_pct - mean_factor_returns) / std_factor_returns
    barr_factor_z_spx_merge = merge_dfs([barr_factor_z_scores, spx_daily.pct_change()])
    barr_factor_z_spx_merge['spx_forward_lag1'] = barr_factor_z_spx_merge['spx'].shift(-1)
    barr_factor_z_spx_merge = barr_factor_z_spx_merge.dropna()
    df = barr_factor_z_spx_merge.copy()

    factor_cols = [c for c in df.columns if c not in ['spx', 'spx_forward_lag1']]
    window_years = 1
    lookback = window_years * 252  # Approx. trading days in 3 years.

    results = []

    for idx in range(lookback, len(df) - 1):  # can't predict for the very last row
        now = df.iloc[idx]
        dt = df.index[idx]

        historical = df.iloc[idx - lookback:idx]
        # Drop NA and any rows in lookback where a factor is missing
        historical = historical.dropna(subset=factor_cols + ['spx_forward_lag1'])

        current_factors = now[factor_cols].values.reshape(1, -1)
        hist_factors = historical[factor_cols].values

        # Use k=5 neighbors
        if len(historical) < 5:
            continue  # Not enough data, skip
        knn = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(hist_factors)
        dist, indices = knn.kneighbors(current_factors)
        neighbor_indices = indices[0]

        pred_spx_next = historical.iloc[neighbor_indices]['spx_forward_lag1'].mean()
        avg_distance = np.mean(dist[0])
        realized_spx_next = df.iloc[idx + 1]['spx']

        results.append({
            'date': dt,
            'knn_pred_spx_avg5': pred_spx_next,
            'knn_avg_distance': avg_distance,
            'realized_spx_next': realized_spx_next
        })

    knn_results = pd.DataFrame(results).set_index('date')

    # Trading logic: long if knn_pred_spx_avg5 > 0, short otherwise
    knn_results['position'] = np.where(knn_results['knn_pred_spx_avg5'] > 0, 1, -1)
    knn_results['strategy_return'] = knn_results['position'] * knn_results['realized_spx_next']

    (knn_results['strategy_return'].mean() * 252) / (knn_results['strategy_return'].std() * (252**0.5))
    (knn_results['realized_spx_next'].mean() * 252) / (knn_results['realized_spx_next'].std() * (252**0.5))


    print(knn_results.head(10))





