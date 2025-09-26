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

def classify_curve_regime(df, short_end='2y', long_end='30y'):
    short = df[short_end]
    long = df[long_end]
    d_short = short.diff()
    d_long = long.diff()
    spread = long - short
    d_spread = spread.diff()

    regimes = []
    for i in range(1, len(df)):
        ds = d_short.iloc[i]
        dl = d_long.iloc[i]
        dsp = d_spread.iloc[i]

        if ds > 0 and dl > 0:  # RATES UP
            regime = 'Bear Steepener' if dsp > 0 else 'Bear Flattener'
        elif ds < 0 and dl < 0:  # RATES DOWN
            regime = 'Bull Steepener' if dsp > 0 else 'Bull Flattener'
        elif abs(dl) >= abs(ds):  # Long end dominant (regardless of sign)
            # Use long end sign for classification
            regime = 'Bear Steepener' if dl > 0 else 'Bull Steepener'
        else:  # Short end dominant
            regime = 'Bear Flattener' if ds > 0 else 'Bull Flattener'
        regimes.append(regime)
    regimes = [None] + regimes
    df['Curve Regime'] = regimes
    return df

def plot_treasury_yield_curves(start,end,**kwargs):
    treasury_yield_curve = pd.DataFrame()
    for each_factor in list(treasury_factors.keys()):
        with open(Path(DATA_DIR) / (each_factor + '.pkl'), 'rb') as file:
            df = pd.read_pickle(file)
        df.index = pd.to_datetime(df.index).values
        df.columns = [treasury_factors[each_factor]]
        treasury_yield_curve = merge_dfs([treasury_yield_curve, df])
    treasury_yield_curve = treasury_yield_curve.dropna()

    classify_curve_regime(treasury_yield_curve,'30y','2y')


    regime_colors = {
        'Bull Steepener': '#E74C3C',  # Reflation (red)
        'Bear Flattener': '#F1C40F',  # Stagflation (yellow)
        'Bear Steepener': '#27AE60',  # Goldilocks (green)
        'Bull Flattener': '#2980B9'  # Deflation (blue)
    }
    treasury_yield_curve['regime_color'] = treasury_yield_curve['Curve Regime'].map(regime_colors)

    ### PLOT ###
    st.title("Equity and Fixed Income by Regime")
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
    df = treasury_yield_curve.copy()

    # Tenors to use (10, omitting '7y' as example)
    subplot_tenors = ['1m', '3m', '6m', '1y', '2y', '3y', '5y', '10y', '20y', '30y']
    regime_colors = {
        'Bear Steepener': '#27AE60',
        'Bear Flattener': '#F1C40F',
        'Bull Steepener': '#E74C3C',
        'Bull Flattener': '#2980B9',
    }
    regime_labels = {
        'Bear Steepener': 'Bear Steepener',
        'Bear Flattener': 'Bear Flattener',
        'Bull Steepener': 'Bull Steepener',
        'Bull Flattener': 'Bull Flattener'
    }

    rows, cols = 2, 5
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[t for t in subplot_tenors],
                        shared_xaxes=True, shared_yaxes=False)

    for i, tenor in enumerate(subplot_tenors):
        r = i // cols + 1
        c = i % cols + 1
        for regime, color in regime_colors.items():
            mask = df['Curve Regime'] == regime
            fig.add_trace(go.Scatter(
                x=df.index[mask],
                y=df[tenor][mask],
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
        title="Treasury Yields by Regime (2x5 Subplots, Tenor by Tenor)",
        legend=dict(title='Regime', x=1.02, y=1),
        margin=dict(t=40, b=40),
        hovermode='closest',
        height=700,
        width=1200
    )
    st.plotly_chart(fig, use_container_width=True)

