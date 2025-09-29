### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- GROWTH INFLATION MODEL ------------------------------------------ ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
import pandas as pd
import functools as ft
import streamlit as st
import plotly.graph_objs as go
from pathlib import Path
import plotly.subplots as sp
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

quad_regime_factors = {
    "SPHB": "high_beta",
    "SPLV": "low_beta",
    "IWM": "small_caps",
    "IWR": "mid_caps",
    "MGK": "mega_cap_growth",
    "IYT": "cyclicals", # or IWN
    "DEF": "defensives",
    "OEF": "size",
    "QUAL": "quality",
    "SPHD": "dividends",
    "MTUM": "momentum",
    "IWD": "value",
    "IWF": "equity_growth",
    "IWB": "large_caps"
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
    total_rows = len(goldilocks_regime) + len(reflation_regime) + len(deflation_regime) + len(stagflation_regime)
    gi_2_factor_results['% of Occurrences'] = [
        len(goldilocks_regime)/ total_rows,
        len(reflation_regime) / total_rows,
        len(deflation_regime) / total_rows,
        len(stagflation_regime) / total_rows
    ]
    gi_2_factor_results['% of Occurrences'] = (gi_2_factor_results['% of Occurrences'] * 100)


    ### ---------------------------------------------------------------------------------------------------------- ###
    ### ------------------------------------------------- PLOTS -------------------------------------------------- ###
    ### ---------------------------------------------------------------------------------------------------------- ###

    ### PLOT ###
    st.title("Growth and Inflation Inputs")
    # Define data columns
    cli_col = 'growth'
    cli_diff_col = 'growth_roc'
    cpi_col = 'inflation'
    cpi_diff_col = 'inflation_roc'

    # Axis/curve settings
    colors = {
        'CLI': '#2056AE',
        'CLI 1st Change': '#6AC47E',
        'CPI': '#F2552C',
        'CPI 1st Change': '#F7BC38'
    }

    # Set up two subplots, both using secondary y-axis
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['CLI (Outright + 1st Order Change)', 'CPI (Outright + 1st Order Change)'],
        specs=[[{"secondary_y": True}, {"secondary_y": True}]]
    )

    # CLI: Outright (primary y), 1st change (secondary y)
    fig.add_trace(
        go.Scatter(
            x=growth_inflation_df.index,
            y=growth_inflation_df[cli_col],
            name='CLI Outright',
            mode='lines',
            line=dict(color=colors['CLI'], width=2)
        ),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=growth_inflation_df.index,
            y=growth_inflation_df[cli_diff_col],
            name='CLI 1st Order Change',
            mode='lines',
            line=dict(color=colors['CLI 1st Change'], dash='dot', width=2)
        ),
        row=1, col=1, secondary_y=True
    )

    # CPI: Outright (primary y), 1st change (secondary y)
    fig.add_trace(
        go.Scatter(
            x=growth_inflation_df.index,
            y=growth_inflation_df[cpi_col],
            name='CPI Outright',
            mode='lines',
            line=dict(color=colors['CPI'], width=2)
        ),
        row=1, col=2, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=growth_inflation_df.index,
            y=growth_inflation_df[cpi_diff_col],
            name='CPI 1st Order Change',
            mode='lines',
            line=dict(color=colors['CPI 1st Change'], dash='dot', width=2)
        ),
        row=1, col=2, secondary_y=True
    )

    fig.update_layout(
        title='Growth (CLI) and Inflation (CPI): Outright & 1st Order Change',
        height=500,
        width=1100,
        hovermode='x unified',
        legend=dict(title='Series', orientation='h', y=-0.2),
        margin=dict(t=50, b=50)
    )
    fig.update_yaxes(title_text="CLI Outright", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="CLI 1st Order Change", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="CPI Outright", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="CPI 1st Order Change", row=1, col=2, secondary_y=True)

    # Streamlit plot
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
    cmap = LinearSegmentedColormap.from_list('red_white_green', ['#ff3333', '#ffffff', '#39b241'], N=256)
    styled = gi_2_factor_results.style \
        .format({'Equities': "{:.2f}%",
                 'Bonds': "{:.2f}%",
                 '% of Occurrences': "{:.2f}%"}) \
        .set_properties(subset=['Equities', 'Bonds', '% of Occurrences'], **{'width': '80px'}) \
        .background_gradient(cmap=cmap, subset=['Equities', 'Bonds'])
    st.title("Growth and Inflation Historical Performance")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write(styled, unsafe_allow_html=True)

    regimes = [
        'Reflation',
        'Stagflation',
        'Goldilocks',
        'Deflation'
    ]
    regime_colors = {
        "Reflation": "#27AE60",
        "Stagflation": "#E74C3C",
        "Goldilocks": "#F1C40F",
        "Deflation": "#2980B9"
    }

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
                x=subdata['sp500_pct'].dropna(),
                name=regime,
                marker_color=regime_colors.get(regime, "#AAAAAA"),
                opacity=0.8,
                nbinsx=30
            ),
            row=row,
            col=col
        )

    # You can set axis titles for all subplots for clarity
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(title_text="SPX % Return", row=row, col=col)
            fig.update_yaxes(title_text="Count", row=row, col=col)

    fig.update_layout(
        title="Distribution of Equity Returns by Regime",
        showlegend=False,
        height=600
    )

    st.title("Equity Return Distributions")
    st.plotly_chart(fig, use_container_width=True)

    ### ---------------------------------------------------------------------------------------------------------- ###
    ### ------------------------------------------------ SECTORS ------------------------------------------------- ###
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

    def highlight_red_green(val):
        if val < 0:
            color = 'background-color: #ffcccc'  # light red
        elif val > 0:
            color = 'background-color: #ccffcc'  # light green
        else:
            color = ''  # no highlight for zero
        return color

    def style_percent(df):
        col = df.columns[0]
        return df.style.format({col: "{:.2f}%"}) \
            .applymap(highlight_red_green, subset=[col])

    ### PLOT ###
    st.title("Top Bottom SPX Sector Performance")
    cols = st.columns(4)
    with cols[0]:
        st.write(style_percent(goldilocks_sector_averages), unsafe_allow_html=True)
    with cols[1]:
        st.write(style_percent(reflation_sector_averages), unsafe_allow_html=True)
    with cols[2]:
        st.write(style_percent(deflation_sector_averages), unsafe_allow_html=True)
    with cols[3]:
        st.write(style_percent(stagflation_sector_averages), unsafe_allow_html=True)

    ### ---------------------------------------------------------------------------------------------------------- ###
    ### ------------------------------------------------ SECTORS ------------------------------------------------- ###
    ### ---------------------------------------------------------------------------------------------------------- ###

    all_quad_regime_factors = pd.DataFrame()
    for each_factor in list(quad_regime_factors.keys()):
        with open(Path(DATA_DIR) / (each_factor + '.csv'), 'rb') as file:
            df = pd.read_csv(file)
        df.index = pd.to_datetime(df['Date']).values
        final_df = pd.DataFrame(df['Close'])
        final_df.columns = [quad_regime_factors[each_factor]]
        all_quad_regime_factors = merge_dfs([all_quad_regime_factors, final_df])

    factors_pct = all_quad_regime_factors.resample('ME').last().pct_change()

    growth_inflation_factors = merge_dfs([growth_inflation_df, factors_pct])

    reflation_factor_regime = growth_inflation_factors[
        (growth_inflation_factors['inflation_roc'] > 0) &
        (growth_inflation_factors['growth_roc'] > 0)
        ]
    stagflation_factor_regime = growth_inflation_factors[
        (growth_inflation_factors['inflation_roc'] > 0) &
        (growth_inflation_factors['growth_roc'] < 0)
        ]
    goldilocks_factor_regime = growth_inflation_factors[
        (growth_inflation_factors['inflation_roc'] < 0) &
        (growth_inflation_factors['growth_roc'] > 0)
        ]
    deflation_factor_regime = growth_inflation_factors[
        (growth_inflation_factors['inflation_roc'] < 0) &
        (growth_inflation_factors['growth_roc'] < 0)
        ]

    reflation_factor_averages = pd.DataFrame((reflation_factor_regime[
                                                  factors_pct.columns].mean(axis=0).sort_values(
        ascending=False) * 100).round(2))
    reflation_factor_averages.columns = ['Reflation']
    stagflation_factor_averages = pd.DataFrame((stagflation_factor_regime[
                                                    factors_pct.columns].mean(axis=0).sort_values(
        ascending=False) * 100).round(2))
    stagflation_factor_averages.columns = ['Stagflation']
    goldilocks_factor_averages = pd.DataFrame((goldilocks_factor_regime[
                                                   factors_pct.columns].mean(axis=0).sort_values(
        ascending=False) * 100).round(2))
    goldilocks_factor_averages.columns = ['Goldilocks']
    deflation_factor_averages = pd.DataFrame((deflation_factor_regime[
                                                  factors_pct.columns].mean(axis=0).sort_values(
        ascending=False) * 100).round(2))
    deflation_factor_averages.columns = ['Deflation']

    # Custom diverging colormap: red (neg), white (zero), green (pos)
    cmap = LinearSegmentedColormap.from_list("red_white_green", ["#ff3333", "#ffffff", "#33cc33"])

    def highlight_red_green(val):
        if val < 0:
            color = 'background-color: #ffcccc'  # light red
        elif val > 0:
            color = 'background-color: #ccffcc'  # light green
        else:
            color = ''  # no highlight for zero
        return color

    def style_percent(df):
        col = df.columns[0]
        return df.style.format({col: "{:.2f}%"}) \
            .applymap(highlight_red_green, subset=[col])

    ### PLOT ###
    st.title("Top Bottom SPX Factor Performance")
    cols = st.columns(4)
    with cols[0]:
        st.write(style_percent(goldilocks_factor_averages), unsafe_allow_html=True)
    with cols[1]:
        st.write(style_percent(reflation_factor_averages), unsafe_allow_html=True)
    with cols[2]:
        st.write(style_percent(deflation_factor_averages), unsafe_allow_html=True)
    with cols[3]:
        st.write(style_percent(stagflation_factor_averages), unsafe_allow_html=True)







