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
import plotly.express as px
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

def classify_regime(short_change, long_change):
    if short_change > 0 and long_change > 0:
        if long_change > short_change:
            return 'Bear Steepening'
        else:
            return 'Bear Flattening'
    elif short_change < 0 and long_change < 0:
        if long_change < short_change:
            return 'Bull Flattening'
        else:
            return 'Bull Steepening'
    elif short_change > 0 and long_change < 0:
        return 'Bear Flattening'
    elif short_change < 0 and long_change > 0:
        return 'Bull Steepening'

def plot_treasury_yield_curves(start,end,**kwargs):
    treasury_yield_curve = pd.DataFrame()
    for each_factor in list(treasury_factors.keys()):
        with open(Path(DATA_DIR) / (each_factor + '.pkl'), 'rb') as file:
            df = pd.read_pickle(file)
        df.index = pd.to_datetime(df.index).values
        df.columns = [treasury_factors[each_factor]]
        treasury_yield_curve = merge_dfs([treasury_yield_curve, df])

    treasury_yield_curve = treasury_yield_curve.resample('ME').last()
    treasury_yield_curve = treasury_yield_curve.dropna()
    treasury_yield_curve['short_chg'] = treasury_yield_curve['2y'].diff()
    treasury_yield_curve['long_chg'] = treasury_yield_curve['10y'].diff()
    treasury_yield_curve['curve_regime'] = np.vectorize(classify_regime)(treasury_yield_curve['short_chg'],
                                                       treasury_yield_curve['long_chg'])

    regime_colors = {
        'Bull Steepening': '#E74C3C',  # Reflation (red)
        'Bear Flattening': '#F1C40F',  # Stagflation (yellow)
        'Bear Steepening': '#27AE60',  # Goldilocks (green)
        'Bull Flattening': '#2980B9'  # Deflation (blue)
    }
    treasury_yield_curve['regime_color'] = treasury_yield_curve['curve_regime'].map(regime_colors)

    ### PLOT ###
    subplot_tenors = ['1m', '3m', '6m', '1y', '2y', '3y', '5y', '10y', '30y']
    regime_labels = {
        'Bear Steepening': 'Bear Steepening',
        'Bear Flattening': 'Bear Flattening',
        'Bull Steepening': 'Bull Steepening',
        'Bull Flattening': 'Bull Flattening'
    }

    rows, cols = 3, 3
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[t for t in subplot_tenors],
                        shared_xaxes=True, shared_yaxes=False)

    for i, tenor in enumerate(subplot_tenors):
        r = i // cols + 1
        c = i % cols + 1
        for regime, color in regime_colors.items():
            mask = treasury_yield_curve['curve_regime'] == regime
            fig.add_trace(go.Scatter(
                x=treasury_yield_curve.index[mask],
                y=treasury_yield_curve[tenor][mask],
                mode='markers',
                marker=dict(size=7, color=color),
                name=regime if (i == 0) else None,  # legend only in first plot
                showlegend=(i == 0),
                hovertemplate=(
                    f"Regime: {regime_labels[regime]}<br>Tenor: {tenor}<br>Yield: %{{y}}<br>Date: %{{x}}"
                ),
                text=[regime_labels[regime]] * mask.sum()
            ), row=r, col=c)

    fig.update_layout(
        legend=dict(title='Regime', x=1.02, y=1),
        margin=dict(t=40, b=40),
        hovermode='closest',
        height=700,
        width=1200
    )
    st.plotly_chart(fig, use_container_width=True)

    ### ---------------------------------------------------------------------------------------------------------- ###
    ### ---------------------------------------------- FIRV FACTORS ---------------------------------------------- ###
    ### ---------------------------------------------------------------------------------------------------------- ###

    ### PULL SPX DATA ###
    with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
        sp500 = pd.read_csv(file)
    sp500.index = pd.to_datetime(sp500['Date']).values
    sp500.drop('Date', axis=1, inplace=True)
    spx_daily = pd.DataFrame(sp500['Close'])
    spx_daily.columns = ['spx']
    spx_daily = spx_daily.resample('ME').last()
    treasury_yield_curve_spx = merge_dfs([treasury_yield_curve,spx_daily]).dropna()
    treasury_yield_curve_spx['spx_pct'] = treasury_yield_curve_spx['spx'].pct_change()

    ### REGIMES ###
    bull_steepening_regime = treasury_yield_curve_spx[(treasury_yield_curve_spx['curve_regime'] == 'Bull Steepening')]
    bull_flattening_regime = treasury_yield_curve_spx[(treasury_yield_curve_spx['curve_regime'] == 'Bull Flattening')]
    bear_steepening_regime = treasury_yield_curve_spx[(treasury_yield_curve_spx['curve_regime'] == 'Bear Steepening')]
    bear_flattening_regime = treasury_yield_curve_spx[(treasury_yield_curve_spx['curve_regime'] == 'Bear Flattening')]

    df = treasury_yield_curve_spx.reset_index().rename(columns={'index': 'date'})
    fig = px.scatter(
        df,
        x='date',
        y='spx',
        color='regime_color',  # for legend (won't show colors directly)
        color_discrete_map='identity',  # interpret regime_color as actual color values
        custom_data=['regime_color', 'spx_pct'],
    )
    fig.update_traces(
        marker=dict(size=8),
        marker_color=df['regime_color'],  # sets actual marker colors per row
    )
    fig.update_layout(
        title='SPX by Regime',
        xaxis_title='Date',
        yaxis_title='SPX Level',
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    ### RESULTS ###
    bull_steepening_regime['spx_pct'].mean()
    bull_flattening_regime['spx_pct'].mean()
    bear_steepening_regime['spx_pct'].mean()
    bear_flattening_regime['spx_pct'].mean()





