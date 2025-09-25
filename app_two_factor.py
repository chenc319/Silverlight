### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- GROWTH INFLATION MODEL ------------------------------------------ ###
### ---------------------------------------------------------------------------------------------------------- ###

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

def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.shift(1).rolling(window=window, min_periods=window).mean()
    rolling_std = series.shift(1).rolling(window=window, min_periods=window).std()
    zscore = (series - rolling_mean) / rolling_std
    return zscore

colors = {
    'securities_outright': '#5FB3FF',   # Vivid sky blue (QE, stable)
    'lending_portfolio':   '#2DCDB2',   # Bright teal/mint (portfolio)
    'treasuries':          '#FFC145',   # Sun gold (Treasury)
    'mbs':                 '#FF6969',   # Approachable coral (MBS)
    'permanent_lending':   '#54C6EB',   # Aqua blue (permanent lending)
    'temporary_lending':   '#FFD166',   # Citrus yellow-orange (temp lending)
    'srf':                 '#6FE7DD',   # Lively turquoise (repo facility)
    'discount_window':     '#8D8DFF',   # Periwinkle (DW lending)
    'fx_swap_line':        '#A685E2',   # Pleasant purple (FX swaps)
    'ppp':                 '#FF8FAB',   # Bright pink (PPP)
    'ms':                  '#FFA952',   # Peach (Main Street)
}

### ---------------------------------------------------------------------------------------------------------- ###
### ------------------------------------------------ DATA PULL ----------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

start = '1900-01-01'
end = pd.to_datetime('today')
def plot_growth_inflation(start, end, **kwargs):
    ### DATA PULL ###
    with open(Path(DATA_DIR) / 'sp500.csv', 'rb') as file:
        sp500 = pd.read_csv(file)
    sp500.index = pd.to_datetime(sp500['Date']).values
    sp500.drop('Date', axis=1, inplace=True)
    sp500.columns = ['close']
    sp500 = sp500.resample('ME').last()

    with open(Path(DATA_DIR) / 'AGG.csv', 'rb') as file:
        agg = pd.read_csv(file)
    agg.index = pd.to_datetime(agg['Date']).values
    agg = pd.DataFrame(agg['Close']).resample('ME').last()

    ### PREPARE FEATURES ###
    growth = pdr.DataReader('USALOLITOAASTSAM', 'fred', start, end).resample('ME').last()
    inflation = pdr.DataReader('CPIAUCSL', 'fred', start, end).resample('ME').last()

    growth_inflation_df = merge_dfs([growth,inflation,sp500,agg]).dropna()
    growth_inflation_df.columns = ['growth','inflation','sp500','bonds']
    growth_inflation_df['growth_roc'] = growth_inflation_df['growth'].diff()
    growth_inflation_df['growth_roc_2'] = growth_inflation_df['growth_roc'].diff()
    growth_inflation_df['inflation_roc'] = growth_inflation_df['inflation'].diff()
    growth_inflation_df['inflation_roc_2'] = growth_inflation_df['inflation_roc'].diff()
    growth_inflation_df['sp500_pct'] = growth_inflation_df['sp500'].pct_change()
    growth_inflation_df['bonds_pct'] = growth_inflation_df['bonds'].pct_change()
    growth_inflation_df = growth_inflation_df.dropna()

    ### ---------------------------------------------------------------------------------------------------------- ###
    ### ------------------------------------------------ ANALYSIS ------------------------------------------------ ###
    ### ---------------------------------------------------------------------------------------------------------- ###

    reflation_regime = growth_inflation_df[
        (growth_inflation_df['inflation_roc'] > 0) &
        (growth_inflation_df['growth_roc'] > 0)
    ]
    stagflation_regime = growth_inflation_df[
        (growth_inflation_df['inflation_roc'] > 0) &
        (growth_inflation_df['growth_roc'] < 0)
    ]
    goldilocks_regime = growth_inflation_df[
        (growth_inflation_df['inflation_roc'] < 0) &
        (growth_inflation_df['growth_roc'] > 0)
    ]
    deflation_regime = growth_inflation_df[
        (growth_inflation_df['inflation_roc'] < 0) &
        (growth_inflation_df['growth_roc'] < 0)
    ]

    def regime_label(row):
        if row['inflation_roc'] > 0 and row['growth_roc'] > 0:
            return 0  # Reflation
        elif row['inflation_roc'] > 0 and row['growth_roc'] < 0:
            return 1  # Stagflation
        elif row['inflation_roc'] < 0 and row['growth_roc'] > 0:
            return 2  # Goldilocks
        elif row['inflation_roc'] < 0 and row['growth_roc'] < 0:
            return 3  # Deflation
        else:
            return np.nan

    growth_inflation_df['regime_code'] = growth_inflation_df.apply(regime_label, axis=1)
    regime_colors = {
        0: '#E74C3C',  # Reflation (red)
        1: '#F1C40F',  # Stagflation (yellow)
        2: '#27AE60',  # Goldilocks (green)
        3: '#2980B9'  # Deflation (blue)
    }

    df = growth_inflation_df.copy()
    df = df.dropna(subset=['regime_code']).copy()
    df = df[~df.index.duplicated(keep='first')]
    df['regime_code'] = df['regime_code'].astype(int)
    df['regime_color'] = df['regime_code'].map(regime_colors)

    reflation_averages = reflation_regime[['sp500_pct','bonds_pct']].mean(axis=0) * 100
    stagflation_averages = stagflation_regime[['sp500_pct','bonds_pct']].mean(axis=0)*  100
    goldilocks_averages = goldilocks_regime[['sp500_pct','bonds_pct']].mean(axis=0) * 100
    deflation_averages = deflation_regime[['sp500_pct','bonds_pct']].mean(axis=0) * 100

    gi_2_factor_results = pd.DataFrame()
    gi_2_factor_results['equities'] = [goldilocks_averages[0],
                                       reflation_averages[0],
                                       deflation_averages[0],
                                       stagflation_averages[0],]
    gi_2_factor_results['bonds'] = [goldilocks_averages[1],
                                    reflation_averages[1],
                                    deflation_averages[1],
                                    stagflation_averages[1]]

    gi_2_factor_results.index =  [
        'Goldilocks (I-G+)',
        'Reflation (I+G+)',
        'Deflation (I-G-)',
        'Stagflation (I+G-)',
    ]

    ### PLOT ###
    cols = ['growth', 'inflation','growth_roc','inflation_roc','growth_roc_2','inflation_roc_2']
    labels = [
        'CLI Outright',
        'CPI Outright',
        'CLI 1st Order Change',
        'CPI 1st Order Change',
        'CPI 2nd Order Change',
        'CLI 2nd Order Change'
    ]
    colors = [
        '#2056AE',  # CLI Outright: rich blue
        '#F2552C',  # CPI Outright: strong orange-red
        '#6AC47E',  # CLI 1st Order Change: fresh green
        '#F7BC38',  # CPI 1st Order Change: gold yellow
        '#AB68D7',  # CPI 2nd Order Change: violet-purple
        '#38C8E7'  # CLI 2nd Order Change: clear aqua blue
    ]
    fig = make_subplots(rows=3, cols=2, subplot_titles=labels)
    for i, (col, color, label) in enumerate(zip(cols, colors, labels)):
        row = i // 2 + 1
        col_position = i % 2 + 1
        fig.add_trace(
            go.Scatter(
                x=growth_inflation_df.index,
                y=growth_inflation_df[col],
                mode='lines',
                name=label,
                line=dict(color=color)
            ),
            row=row,
            col=col_position
        )
    fig.update_layout(
        title="Growth and Inflation Factors",
        showlegend=False,
        height=900,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    ### PLOT ###
    regime_colors = {
        0: '#E74C3C',  # Reflation (red)
        1: '#F1C40F',  # Stagflation (yellow)
        2: '#27AE60',  # Goldilocks (green)
        3: '#2980B9'  # Deflation (blue)
    }
    regime_labels = {
        0: 'Reflation',
        1: 'Stagflation',
        2: 'Goldilocks',
        3: 'Deflation'
    }
    df = growth_inflation_df.copy()
    df = df.dropna(subset=['regime_code']).copy()
    df = df[~df.index.duplicated(keep='first')]
    df['regime_code'] = df['regime_code'].astype(int)
    df['regime_color'] = df['regime_code'].map(regime_colors)
    df['regime_label'] = df['regime_code'].map(regime_labels)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['sp500'],
        mode='lines',
        line=dict(color='black', width=2),
        name='SP500',
        showlegend=False,
        hoverinfo='skip'  # Suppress hover for line
    ))
    for code, color in regime_colors.items():
        mask = df['regime_code'] == code
        fig.add_trace(go.Scatter(
            x=df.index[mask],
            y=df['sp500'][mask],
            mode='markers',
            marker=dict(color=color, size=8),
            name=regime_labels[code],
            showlegend=True,
            hovertemplate=(
                "Regime: %{text}<br>SP500: %{y}<br>Date: %{x}"
            ),
            text=[regime_labels[code]] * sum(mask)
        ))
    fig.update_layout(
        title="SP500 by Regime",
        hovermode='x',  # Only show one trace at a time
        legend=dict(title='Regime')
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['bonds'],
        mode='lines',
        line=dict(color='black', width=2),
        name='Bonds',
        showlegend=False,
        hoverinfo='skip'  # Suppress hover for line
    ))
    for code, color in regime_colors.items():
        mask = df['regime_code'] == code
        fig.add_trace(go.Scatter(
            x=df.index[mask],
            y=df['bonds'][mask],
            mode='markers',
            marker=dict(color=color, size=8),
            name=regime_labels[code],
            showlegend=True,
            hovertemplate=(
                "Regime: %{text}<br>SP500: %{y}<br>Date: %{x}"
            ),
            text=[regime_labels[code]] * sum(mask)
        ))
    fig.update_layout(
        title="Bonds by Regime",
        hovermode='x',  # Only show one trace at a time
        legend=dict(title='Regime')
    )
    st.plotly_chart(fig, use_container_width=True)

    from matplotlib.colors import LinearSegmentedColormap

    # Create a colormap from red (neg), white (zero), to green (pos)
    cmap = LinearSegmentedColormap.from_list('red_white_green', [
        "#ff3333", "#ffffff", "#39b241"
    ], N=256)

    styled = gi_2_factor_results.style.background_gradient(cmap=cmap, subset=['bonds'])
    st.dataframe(styled, use_container_width=True, hide_index=True)





