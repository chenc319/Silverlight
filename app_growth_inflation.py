### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- GROWTH INFLATION MODEL ------------------------------------------ ###
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

spx_sectors = {
    "XLC": "Comm Services",
    "XLY": "Cons Disc",
    "XLP": "Cons Stap",
    "XLE": "Energy",
    "XLF": "Financial",
    "XLV": "Healthcare",
    "XLI": "Industrial",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLK": "Tech",
    "XLU": "utilities"
}

def merge_dfs(array_of_dfs):
    return ft.reduce(lambda left, right: pd.merge(left, right,
                                                  left_index=True,
                                                  right_index=True, how='outer'), array_of_dfs)

### ---------------------------------------------------------------------------------------------------------- ###
### ------------------------------------------------ DATA PULL ----------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###


def plot_growth_inflation(start, end, **kwargs):
    ### DATA PULL ###
    with open(Path(DATA_DIR) / 'sp500.csv', 'rb') as file:
        sp500 = pd.read_csv(file)
    with open(Path(DATA_DIR) / 'growth.pkl', 'rb') as file:
        growth = pd.read_pickle(file)
    with open(Path(DATA_DIR) / 'inflation.pkl', 'rb') as file:
        inflation = pd.read_pickle(file)
    with open(Path(DATA_DIR) / 'AGG.csv', 'rb') as file:
        agg = pd.read_csv(file)

    with open(Path(DATA_DIR) / 'XLB.csv', 'rb') as file:
        xlb_df = pd.read_csv(file)
    with open(Path(DATA_DIR) / 'XLC.csv', 'rb') as file:
        xlc_df = pd.read_csv(file)
    with open(Path(DATA_DIR) / 'XLE.csv', 'rb') as file:
        xle_df = pd.read_csv(file)
    with open(Path(DATA_DIR) / 'XLF.csv', 'rb') as file:
        xlf_df = pd.read_csv(file)
    with open(Path(DATA_DIR) / 'XLI.csv', 'rb') as file:
        xli_df = pd.read_csv(file)
    with open(Path(DATA_DIR) / 'XLK.csv', 'rb') as file:
        xlk_df = pd.read_csv(file)
    with open(Path(DATA_DIR) / 'XLP.csv', 'rb') as file:
        xlp_df = pd.read_csv(file)
    with open(Path(DATA_DIR) / 'XLRE.csv', 'rb') as file:
        xlre_df = pd.read_csv(file)
    with open(Path(DATA_DIR) / 'XLU.csv', 'rb') as file:
        xlu_df = pd.read_csv(file)
    with open(Path(DATA_DIR) / 'XLV.csv', 'rb') as file:
        xlv_df = pd.read_csv(file)
    with open(Path(DATA_DIR) / 'XLY.csv', 'rb') as file:
        xly_df = pd.read_csv(file)

    xlb_df.index = pd.to_datetime(xlb_df['Date']).values
    xlb = pd.DataFrame(xlb_df['Close'])
    xlb.columns = ['xlb']
    xlc_df.index = pd.to_datetime(xlc_df['Date']).values
    xlc = pd.DataFrame(xlc_df['Close'])
    xlc.columns = ['xlc']
    xle_df.index = pd.to_datetime(xle_df['Date']).values
    xle = pd.DataFrame(xle_df['Close'])
    xle.columns = ['xle']
    xlf_df.index = pd.to_datetime(xlf_df['Date']).values
    xlf = pd.DataFrame(xlf_df['Close'])
    xlf.columns = ['xlf']
    xli_df.index = pd.to_datetime(xli_df['Date']).values
    xli = pd.DataFrame(xli_df['Close'])
    xli.columns = ['xli']
    xlk_df.index = pd.to_datetime(xlk_df['Date']).values
    xlk = pd.DataFrame(xlk_df['Close'])
    xlk.columns = ['xlk']
    xlp_df.index = pd.to_datetime(xlp_df['Date']).values
    xlp = pd.DataFrame(xlp_df['Close'])
    xlp.columns = ['xlp']
    xlre_df.index = pd.to_datetime(xlre_df['Date']).values
    xlre = pd.DataFrame(xlre_df['Close'])
    xlre.columns = ['xlre']
    xlu_df.index = pd.to_datetime(xlu_df['Date']).values
    xlu = pd.DataFrame(xlu_df['Close'])
    xlu.columns = ['xlu']
    xlv_df.index = pd.to_datetime(xlv_df['Date']).values
    xlv = pd.DataFrame(xlv_df['Close'])
    xlv.columns = ['xlv']
    xly_df.index = pd.to_datetime(xly_df['Date']).values
    xly = pd.DataFrame(xly_df['Close'])
    xly.columns = ['xly']

    sp500.index = pd.to_datetime(sp500['Date']).values
    sp500.drop('Date', axis=1, inplace=True)
    sp500.columns = ['close']
    sp500 = sp500.resample('ME').last()
    agg.index = pd.to_datetime(agg['Date']).values
    agg = pd.DataFrame(agg['Close']).resample('ME').last()

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
    gi_2_factor_results['Regime'] = [
        'Goldilocks (I-G+)',
        'Reflation (I+G+)',
        'Deflation (I-G-)',
        'Stagflation (I+G-)',
    ]
    gi_2_factor_results['Equities'] = [goldilocks_averages[0],
                                       reflation_averages[0],
                                       deflation_averages[0],
                                       stagflation_averages[0],]
    gi_2_factor_results['Bonds'] = [goldilocks_averages[1],
                                    reflation_averages[1],
                                    deflation_averages[1],
                                    stagflation_averages[1]]

    ### ---------------------------------------------------------------------------------------------------------- ###
    ### ------------------------------------------------- PLOTS -------------------------------------------------- ###
    ### ---------------------------------------------------------------------------------------------------------- ###

    ### PLOT ###
    st.title("Growth and Inflation Inputs")
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

    ### TABLE ###
    st.title("Growth and Inflation Backtest Results")
    cmap = LinearSegmentedColormap.from_list('red_white_green', [
        "#ff3333", "#ffffff", "#39b241"
    ], N=256)

    styled = gi_2_factor_results.style \
        .format({'Equities': "{:.2f}%", 'Bonds': "{:.2f}%"}) \
        .set_properties(subset=['Equities', 'Bonds'], **{'width': '80px'}) \
        .background_gradient(cmap=cmap, subset=['Equities', 'Bonds'])
    st.dataframe(styled, use_container_width=False, hide_index=True)

    ### ---------------------------------------------------------------------------------------------------------- ###
    ### ------------------------------------------------- PLOTS -------------------------------------------------- ###
    ### ---------------------------------------------------------------------------------------------------------- ###

    sector_merge = merge_dfs([xlc,xly, xlp,
                              xle, xlf, xlv,
                              xli, xlb, xlre,
                              xlk, xlu]).resample('ME').last().pct_change()
    sector_merge.columns = ['comm_serv','cons_disc', 'cons_stap', 'energy',
                            'financials', 'healthcare', 'industrial', 'materials',
                            'real_estate', 'tech', 'utilities']
    growth_inflation_sector = merge_dfs([growth_inflation_df,sector_merge])

    reflation_sector_regime = growth_inflation_sector[
        (growth_inflation_sector['inflation_roc'] > 0) &
        (growth_inflation_sector['growth_roc'] > 0)
        ]
    stagflation_sector_regime = growth_inflation_sector[
        (growth_inflation_sector['inflation_roc'] > 0) &
        (growth_inflation_sector['growth_roc'] < 0)
        ]
    goldilocks_sector_regime = growth_inflation_sector[
        (growth_inflation_sector['inflation_roc'] < 0) &
        (growth_inflation_sector['growth_roc'] > 0)
        ]
    deflation_sector_regime = growth_inflation_sector[
        (growth_inflation_sector['inflation_roc'] < 0) &
        (growth_inflation_sector['growth_roc'] < 0)
        ]

    reflation_sector_averages = pd.DataFrame((reflation_sector_regime[
        sector_merge.columns].mean(axis=0).sort_values(ascending=False) * 100).round(2))
    reflation_sector_averages.columns = ['Reflation']
    stagflation_sector_averages = pd.DataFrame((stagflation_sector_regime[
        sector_merge.columns].mean(axis=0).sort_values(ascending=False) * 100).round(2))
    stagflation_sector_averages.columns = ['Stagflation']
    goldilocks_sector_averages = pd.DataFrame((goldilocks_sector_regime[
        sector_merge.columns].mean(axis=0).sort_values(ascending=False) * 100).round(2))
    goldilocks_sector_averages.columns = ['Goldilocks']
    deflation_sector_averages = pd.DataFrame((deflation_sector_regime[
        sector_merge.columns].mean(axis=0).sort_values(ascending=False) * 100).round(2))
    deflation_sector_averages.columns = ['Deflation']

    # Custom diverging colormap: red (neg), white (zero), green (pos)
    cmap = LinearSegmentedColormap.from_list("red_white_green", ["#ff3333", "#ffffff", "#33cc33"])

    def style_df(df):
        colname = df.columns[0]
        return df.style.format({colname: "{:.2f}%"}) \
            .background_gradient(cmap=cmap, vmin=df[colname].min(), vmax=df[colname].max(), subset=[colname])

    ### PLOT ###
    st.title("Top Bottom SPX Sector Performance")
    cols = st.columns(4)
    with cols[0]:
        st.write(style_df(reflation_sector_averages), unsafe_allow_html=True)
    with cols[1]:
        st.write(style_df(stagflation_sector_averages), unsafe_allow_html=True)
    with cols[2]:
        st.write(style_df(goldilocks_sector_averages), unsafe_allow_html=True)
    with cols[3]:
        st.write(style_df(deflation_sector_averages), unsafe_allow_html=True)










