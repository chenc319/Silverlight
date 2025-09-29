### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------- FIRV FACTORS ---------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
import pandas as pd
import functools as ft
import streamlit as st
import plotly.graph_objs as go
from pathlib import Path
import plotly.express as px
import os
from matplotlib.colors import LinearSegmentedColormap
from plotly.subplots import make_subplots
import numpy as np
import plotly.subplots as sp
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
with open(Path(DATA_DIR) / 'sp500.csv', 'rb') as file:
    sp500 = pd.read_csv(file)
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

    ### RESULTS ###
    yc_regime_results = pd.DataFrame()
    yc_regime_results['Regime'] = [
        'Bear Flattening',
        'Bull Flattening',
        'Bear Steepening',
        'Bull Steepening'
    ]
    yc_regime_results['SPX'] = [
        bear_flattening_regime['spx_pct'].mean() * 100,
        bull_flattening_regime['spx_pct'].mean() * 100,
        bear_steepening_regime['spx_pct'].mean() * 100,
        bull_steepening_regime['spx_pct'].mean() * 100
    ]
    total_rows = (len(bull_steepening_regime) + len(bull_flattening_regime) +
                  len(bear_steepening_regime) + len(bear_flattening_regime))
    yc_regime_results['% of Occurrences'] = [
        len(bear_flattening_regime) / total_rows,
        len(bull_flattening_regime) / total_rows,
        len(bear_steepening_regime) / total_rows,
        len(bull_steepening_regime) / total_rows
    ]
    yc_regime_results['% of Occurrences'] = (yc_regime_results['% of Occurrences'] * 100)

    ### PLOT ###
    st.title('SPX by YC Regime')
    df = treasury_yield_curve_spx.reset_index().rename(columns={'index': 'date'})
    regime_code_to_label = {
        "#2980B9": "Bull Steepening",
        "#E74C3C": "Bear Steepening",
        "#27AE60": "Bear Flattening",
        "#F1C40F": "Bull Flattening"
    }
    df["regime_label"] = df["regime_color"].map(regime_code_to_label)
    regime_color_map = {
        "Bull Steepening": "#2980B9",
        "Bear Steepening": "#E74C3C",
        "Bear Flattening": "#27AE60",
        "Bull Flattening": "#F1C40F"
    }
    fig = px.scatter(
        df,
        x="date",
        y="spx",
        color="regime_label",
        color_discrete_map=regime_color_map,
        custom_data=["spx_pct"]
    )

    fig.update_traces(
        marker=dict(size=8),
        selector=dict(mode='markers')
    )
    fig.update_layout(
        title='SPX by Yield Curve Regime',
        xaxis_title='Date',
        yaxis_title='SPX Level',
        legend_title='Curve Regime'
    )
    st.plotly_chart(fig, use_container_width=True)

    ### TABLE ###
    st.title("YC Regime SPX Historical Performance")
    cmap = LinearSegmentedColormap.from_list('red_white_green', ['#ff3333', '#ffffff', '#39b241'], N=256)
    styled = yc_regime_results.style \
        .format({'SPX': "{:.2f}%",
                 '% of Occurrences': "{:.2f}%"}) \
        .set_properties(subset=['SPX', '% of Occurrences'], **{'width': '80px'}) \
        .background_gradient(cmap=cmap, subset=['SPX'])
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write(styled, unsafe_allow_html=True)

    ### RETURN DISTRIBUTIONS ###
    st.title("YC Regime SPX Return Distributions")
    regimes = [
        "Bear Flattening",
        "Bear Steepening",
        "Bull Flattening",
        "Bull Steepening",
    ]
    min_bound = df['spx_pct'].min()
    max_bound = df['spx_pct'].max()
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=regimes
    )
    for i, regime in enumerate(regimes):
        row = i // 2 + 1
        col = i % 2 + 1
        subdata = df[df['regime_label'] == regime]
        fig.add_trace(
            go.Histogram(
                x=subdata['spx_pct'].dropna(),
                name=regime,
                marker_color=regime_color_map[regime],
                opacity=0.8,
                nbinsx=30,
                xbins=dict(
                    start=min_bound,
                    end=max_bound
                )
            ),
            row=row,
            col=col
        )
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(title_text="SPX % Return", row=row, col=col, range=[min_bound, max_bound])
            fig.update_yaxes(title_text="Count", row=row, col=col)
    fig.update_layout(
        showlegend=False,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)







