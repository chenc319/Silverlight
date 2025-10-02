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
from sklearn.decomposition import PCA
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
### --------------------------------- ADVANCED GROWTH AND INFLATION FACTORS ---------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

growth_dict = {
    'USALOLITOAASTSAM': 'cli_amplitude_adjusted',
    'INDPRO': 'industrial_production',
    'BOPGSTB': 'trade_balance_goods_and_services',
    'RSXFS': 'advanced_retail_sales_retail_trade',
    'TLMFGCONS': 'manufacturing_spending',
    'PAYEMS': 'all_employees_total_nonfarm',
    'USGOOD': 'goods_producing_employment',
    'MANEMP': 'all_employees_manufacturing',
    'CES0500000011': 'avg_earnings_all_private_employees',
    'PCEC96': 'real_personal_consumption_expenditures',
    'RRSFS': 'real_retail_food_services_sales',
    'TOTALSA': 'total_vehicle_sales'
}
inflation_dict = {
    'CPIAUCSL': 'cpi_all_items',
    'CPILFESL': 'cpi_less_food_energy',
    'PPIACO': 'ppi_all_commodities',
    'CPIUFDSL': 'cpi_food',
    'CPIENGSL': 'cpi_energy',
    'CUSR0000SAH3': 'cpi_household_furnishings',
    'CPIAPPSL': 'cpi_apparel',
    'CPIMEDSL': 'cpi_medical_care',
    'CPITRNSL': 'cpi_transportation',
    'CUSR0000SAF116': 'cpi_alcohol',
    'CUSR0000SETB': 'cpi_motor_fuel',
    'CUSR0000SASLE': 'cpi_services_less_energy'
}
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

### SPX DATA ###
with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
    sp500 = pd.read_csv(file)
sp500.index = pd.to_datetime(sp500['Date']).values
sp500.drop('Date', axis=1, inplace=True)
spx_monthly = pd.DataFrame(sp500['Close']).resample('ME').last()
spx_monthly.columns = ['spx']
spx_monthly_pct = spx_monthly.pct_change().dropna()

### GROWTH AND INFLATION ###
with open(Path(DATA_DIR) / 'grid_growth_variables.pkl', 'rb') as file:
    grid_growth_variables = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'grid_inflation_variables.pkl', 'rb') as file:
    grid_inflation_variables = pd.read_pickle(file)
grid_growth_pct = grid_growth_variables.diff().dropna()
grid_growth_pct.columns = growth_dict.keys()
grid_growth_pct_spx_merge = merge_dfs([grid_growth_pct,spx_monthly_pct.shift(-1)]).dropna()
grid_inflation_pct = grid_inflation_variables.diff().dropna()
grid_inflation_pct.columns = inflation_dict.keys()
grid_inflation_pct_spx_merge = merge_dfs([grid_inflation_pct,spx_monthly_pct.shift(-1)]).dropna()

### GROWTH Z SCORED ###
grid_growth_mean = grid_growth_pct.rolling(12).mean()
grid_growth_std = grid_growth_pct.rolling(12).std()
grid_growth_z = (grid_growth_pct - grid_growth_mean) / grid_growth_std
grid_growth_z.columns = growth_dict.values()

### INFLATION Z SCORED ###
grid_inflation_mean = grid_inflation_pct.rolling(12).mean()
grid_inflation_std = grid_inflation_pct.rolling(12).std()
grid_inflation_z = (grid_inflation_pct - grid_inflation_mean) / grid_inflation_std
grid_inflation_z.columns = inflation_dict.values()

### SPX GROWTH CORRELATION WEIGHTS ###
window_size = 12
grid_growth_corr_spx = pd.DataFrame(index=grid_growth_pct_spx_merge.index)
factor = grid_growth_pct_spx_merge.iloc[:, 12]
for i in range(12):
    grid_growth_corr_spx[grid_growth_pct_spx_merge.columns[i] + '_corr'] = (
        grid_growth_pct_spx_merge.iloc[:, i].rolling(window=window_size).corr(factor))
grid_growth_corr_spx.columns = growth_dict.values()

### SPX INFLATION CORRELATION WEIGHTS ###
window_size = 12
grid_inflation_corr_spx = pd.DataFrame(index=grid_inflation_pct_spx_merge.index)
factor = grid_inflation_pct_spx_merge.iloc[:, 12]
for i in range(12):
    grid_inflation_corr_spx[grid_inflation_pct_spx_merge.columns[i] + '_corr'] = (
        grid_inflation_pct_spx_merge.iloc[:, i].rolling(window=window_size).corr(factor))
grid_inflation_corr_spx.columns = inflation_dict.values()

### SECTOR DATA ###
spx_sectors_merge = pd.DataFrame()
for each_factor in list(spx_sectors.keys()):
    with open(Path(DATA_DIR) / (each_factor + '.csv'), 'rb') as file:
        df = pd.read_csv(file)
    df.index = pd.to_datetime(df['Date']).values
    df = pd.DataFrame(df['Close'])
    df.columns = [spx_sectors[each_factor]]
    spx_sectors_merge = merge_dfs([spx_sectors_merge, df])

def plot_grid_factors(start,end,**kwargs):
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
        "#ffb7eb",  # Bubblegum Pink
        "#78b4a4",  # Soft Sage Green
    ]
    ### PLOT ###
    st.title('GRID Growth Factors')
    df = grid_growth_pct.copy().resample('ME').last()
    columns_to_plot = grid_growth_pct.columns
    fig = sp.make_subplots(rows=3, cols=4, subplot_titles=columns_to_plot)
    for i, col in enumerate(columns_to_plot):
        row = i // 4 + 1
        col_pos = i % 4 + 1
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
        height=800,
        width=1400
    )
    st.plotly_chart(fig, use_container_width=True)

    ### PLOT ###
    st.title('GRID Inflation Factors')
    df = grid_inflation_pct.copy().resample('ME').last()
    columns_to_plot = grid_inflation_pct.columns
    fig = sp.make_subplots(rows=3, cols=4, subplot_titles=columns_to_plot)
    for i, col in enumerate(columns_to_plot):
        row = i // 4 + 1
        col_pos = i % 4 + 1
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
        height=800,
        width=1400
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_factor_correlation_to_lagged_spx(start,end,**kwargs):
    cols = grid_growth_corr_spx.columns
    n = len(cols)
    rows = (n + 2) // 4  # 3 wide grid, adjust for aesthetics
    cols_per_row = 4
    fig = sp.make_subplots(
        rows=rows, cols=cols_per_row,
        subplot_titles=cols,
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )
    color_cycle = ['#27ae60', '#2675b9', '#fa983a',
                   '#8e44ad', '#b33771', '#f9ca24',
                   '#e84118', '#00b894', '#636e72',
                   '#d35400', '#6c5ce7', '#00cec9']
    for i, col in enumerate(cols):
        row = i // cols_per_row + 1
        colpos = i % cols_per_row + 1
        fig.add_trace(go.Scatter(
            x=grid_growth_corr_spx.index,
            y=grid_growth_corr_spx[col],
            mode='lines',
            name=col,
            line=dict(color=color_cycle[i % len(color_cycle)], width=2),
            hovertemplate=f"{col}<br>Date: %{{x}}<br>Rolling Corr: %{{y:.2f}}<extra></extra>"
        ), row=row, col=colpos)
    fig.update_layout(
        height=320 * rows,
        showlegend=False,
        template='plotly_white',
        title=f"Rolling {12}-Month Correlation Growth to SPX",
        margin=dict(l=40, r=30, t=75, b=40)
    )
    for i in range(1, rows + 1):
        for j in range(1, cols_per_row + 1):
            fig.update_yaxes(range=[-1, 1], row=i, col=j)
    st.plotly_chart(fig, use_container_width=True)

    cols = grid_inflation_corr_spx.columns
    n = len(cols)
    rows = (n + 2) // 4  # 3 wide grid, adjust for aesthetics
    cols_per_row = 4
    fig = sp.make_subplots(
        rows=rows, cols=cols_per_row,
        subplot_titles=cols,
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )
    color_cycle = ['#27ae60', '#2675b9', '#fa983a',
                   '#8e44ad', '#b33771', '#f9ca24',
                   '#e84118', '#00b894', '#636e72',
                   '#d35400', '#6c5ce7', '#00cec9']
    for i, col in enumerate(cols):
        row = i // cols_per_row + 1
        colpos = i % cols_per_row + 1
        fig.add_trace(go.Scatter(
            x=grid_inflation_corr_spx.index,
            y=grid_inflation_corr_spx[col],
            mode='lines',
            name=col,
            line=dict(color=color_cycle[i % len(color_cycle)], width=2),
            hovertemplate=f"{col}<br>Date: %{{x}}<br>Rolling Corr: %{{y:.2f}}<extra></extra>"
        ), row=row, col=colpos)
    fig.update_layout(
        height=320 * rows,
        showlegend=False,
        template='plotly_white',
        title=f"Rolling {12}-Month Correlation Inflation to SPX",
        margin=dict(l=40, r=30, t=75, b=40)
    )
    for i in range(1, rows + 1):
        for j in range(1, cols_per_row + 1):
            fig.update_yaxes(range=[-1, 1], row=i, col=j)
    st.plotly_chart(fig, use_container_width=True)

def plot_grid_factors_regime_performance(start, end, **kwargs):
    grid_growth_cross_mean_z = pd.DataFrame(grid_growth_z * grid_growth_corr_spx).mean(axis=1)
    grid_inflation_cross_mean_z = pd.DataFrame(grid_inflation_z * grid_inflation_corr_spx).mean(axis=1)
    grid_growth_inflation_spx = pd.concat([
        grid_growth_cross_mean_z,
        grid_inflation_cross_mean_z,
        spx_monthly_pct.shift(-1)
    ], axis=1).dropna()
    grid_growth_inflation_spx.columns = ['growth', 'inflation', 'spx']

    def regime_label(row):
        if row['inflation'] > 0 and row['growth'] > 0:
            return 0  # Reflation
        elif row['inflation'] > 0 and row['growth'] < 0:
            return 1  # Stagflation
        elif row['inflation'] < 0 and row['growth'] > 0:
            return 2  # Goldilocks
        elif row['inflation'] < 0 and row['growth'] < 0:
            return 3  # Deflation
        else:
            return np.nan

    df = grid_growth_inflation_spx.copy()
    df['regime_code'] = df.apply(regime_label, axis=1)
    df = df.dropna(subset=['regime_code']).copy()
    df['regime_code'] = df['regime_code'].astype(int)
    regime_labels = {
        0: 'Reflation',
        1: 'Stagflation',
        2: 'Goldilocks',
        3: 'Deflation'
    }
    regime_colors = {
        0: '#E74C3C',  # Reflation (red)
        1: '#F1C40F',  # Stagflation (yellow)
        2: '#27AE60',  # Goldilocks (green)
        3: '#2980B9'  # Deflation (blue)
    }
    df['regime_color'] = df['regime_code'].map(regime_colors)
    df['regime_label'] = df['regime_code'].map(regime_labels)

    reflation_regime = df[df['regime_label'] == 'Reflation']
    stagflation_regime = df[df['regime_label'] == 'Stagflation']
    goldilocks_regime = df[df['regime_label'] == 'Goldilocks']
    deflation_regime = df[df['regime_label'] == 'Deflation']

    reflation_mean_return = reflation_regime['spx'].mean() * 100
    stagflation_mean_return = stagflation_regime['spx'].mean() * 100
    goldilocks_mean_return = goldilocks_regime['spx'].mean() * 100
    deflation_mean_return = deflation_regime['spx'].mean() * 100

    reflation_ann_return = reflation_regime['spx'].mean() * 12 * 100
    stagflation_ann_return = stagflation_regime['spx'].mean() * 12 * 100
    goldilocks_ann_return = goldilocks_regime['spx'].mean() * 12 * 100
    deflation_ann_return = deflation_regime['spx'].mean() * 12 * 100

    reflation_volatility = reflation_regime['spx'].std() * (12 ** 0.5) * 100
    stagflation_volatility = stagflation_regime['spx'].std() * (12 ** 0.5) * 100
    goldilocks_volatility = goldilocks_regime['spx'].std() * (12 ** 0.5) * 100
    deflation_volatility = deflation_regime['spx'].std() * (12 ** 0.5) * 100

    reflation_win_ratio = reflation_regime[reflation_regime['spx'] > 0].shape[0] / reflation_regime.shape[0] * 100
    stagflation_win_ratio = stagflation_regime[stagflation_regime['spx'] > 0].shape[0] / stagflation_regime.shape[
        0] * 100
    goldilocks_win_ratio = goldilocks_regime[goldilocks_regime['spx'] > 0].shape[0] / goldilocks_regime.shape[0] * 100
    deflation_win_ratio = deflation_regime[deflation_regime['spx'] > 0].shape[0] / deflation_regime.shape[0] * 100

    total_rows = sum(
        [goldilocks_regime.shape[0], reflation_regime.shape[0], deflation_regime.shape[0], stagflation_regime.shape[0]])
    grid_results = pd.DataFrame()
    grid_results['Regime'] = [
        'Goldilocks (I-G+)',
        'Reflation (I+G+)',
        'Deflation (I-G-)',
        'Stagflation (I+G-)',
    ]
    grid_results['Mean Monthly Returns'] = [
        goldilocks_mean_return,
        reflation_mean_return,
        deflation_mean_return,
        stagflation_mean_return
    ]
    grid_results['Ann. Returns'] = [
        goldilocks_ann_return,
        reflation_ann_return,
        deflation_ann_return,
        stagflation_ann_return
    ]
    grid_results['Ann. Volatility'] = [
        goldilocks_volatility,
        reflation_volatility,
        deflation_volatility,
        stagflation_volatility
    ]
    grid_results['Return/Risk'] = grid_results['Ann. Returns'] / grid_results['Ann. Volatility']
    grid_results['Win Ratio'] = [
        goldilocks_win_ratio,
        reflation_win_ratio,
        deflation_win_ratio,
        stagflation_win_ratio
    ]
    grid_results['% of Occurrences'] = [
        goldilocks_regime.shape[0] / total_rows * 100,
        reflation_regime.shape[0] / total_rows * 100,
        deflation_regime.shape[0] / total_rows * 100,
        stagflation_regime.shape[0] / total_rows * 100
    ]

    st.title("GRID Regime Performance")
    cmap = LinearSegmentedColormap.from_list('red_white_green', ['#ff3333', '#ffffff', '#39b241'], N=256)
    styled = grid_results.style \
        .format({
        'Mean Monthly Returns': "{:.2f}%",
        'Ann. Returns': "{:.2f}%",
        'Ann. Volatility': "{:.2f}%",
        'Return/Risk': "{:.2f}",
        'Win Ratio': "{:.2f}%",
        '% of Occurrences': "{:.2f}%"
    }) \
        .set_properties(
        subset=['Mean Monthly Returns', 'Ann. Returns', 'Ann. Volatility', 'Return/Risk', 'Win Ratio'],
        **{'width': '500px'}
    ) \
        .background_gradient(cmap=cmap, subset=[
        'Mean Monthly Returns', 'Ann. Returns', 'Ann. Volatility', 'Return/Risk', 'Win Ratio'
    ])

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write(styled, unsafe_allow_html=True)

    # --- Bonds Return Distribution ---
    st.title("GRID Return Distributions")
    regimes = ['Reflation', 'Stagflation', 'Goldilocks', 'Deflation']
    regime_colors_plotly = {
        "Reflation": "#E74C3C",
        "Stagflation": "#F1C40F",
        "Goldilocks": "#27AE60",
        "Deflation": "#2980B9"
    }

    fig = make_subplots(rows=2, cols=2, subplot_titles=regimes)
    min_bound = df['spx'].min()
    max_bound = df['spx'].max()
    for i, regime in enumerate(regimes):
        row = i // 2 + 1
        col = i % 2 + 1
        subdata = df[df['regime_label'] == regime]
        fig.add_trace(
            go.Histogram(
                x=subdata['spx'].dropna(),
                name=regime,
                marker_color=regime_colors_plotly.get(regime, "#AAAAAA"),
                opacity=0.8,
                nbinsx=30
            ),
            row=row,
            col=col
        )
        fig.update_xaxes(title_text="Equity % Return", row=row, col=col, range=[min_bound, max_bound])
        fig.update_yaxes(title_text="Count", row=row, col=col)
    fig.update_layout(showlegend=False, height=600)
    st.plotly_chart(fig, use_container_width=True)

def grid_z_score_backtest(start, end, **kwargs):
    grid_growth_cross_mean_z = pd.DataFrame(grid_growth_z * grid_growth_corr_spx).mean(axis=1)
    grid_inflation_cross_mean_z = pd.DataFrame(grid_inflation_z* grid_inflation_corr_spx).mean(axis=1)
    grid_growth_inflation_spx = pd.concat([
        grid_growth_cross_mean_z,
        grid_inflation_cross_mean_z,
        spx_monthly_pct.shift(-1)
    ], axis=1).dropna()
    grid_growth_inflation_spx.columns = ['growth', 'inflation', 'spx']

    def regime_label(row):
        if row['inflation'] > 0 and row['growth'] > 0:
            return 0  # Reflation
        elif row['inflation'] > 0 and row['growth'] < 0:
            return 1  # Stagflation
        elif row['inflation'] < 0 and row['growth'] > 0:
            return 2  # Goldilocks
        elif row['inflation'] < 0 and row['growth'] < 0:
            return 3  # Deflation
        else:
            return np.nan
    regime_labels = {
        0: 'Reflation',
        1: 'Stagflation',
        2: 'Goldilocks',
        3: 'Deflation'
    }
    grid_growth_inflation_spx['regime_code'] = grid_growth_inflation_spx.apply(regime_label, axis=1)
    grid_growth_inflation_spx['regime_label'] = grid_growth_inflation_spx['regime_code'].map(regime_labels)

    def grid_backtest(row):
        if row['regime_label']== 'Goldilocks':
            return 1
        elif row['regime_label']== 'Reflation':
            return 0.75
        elif row['regime_label']== 'Deflation':
            return 0.50
        elif row['regime_label']== 'Stagflation':
            return 0.25
        else:
            return np.nan

    grid_growth_inflation_spx['weights'] = grid_growth_inflation_spx.apply(grid_backtest, axis=1)
    grid_growth_inflation_spx['bt_returns'] = grid_growth_inflation_spx['weights'] * grid_growth_inflation_spx['spx']
    grid_growth_inflation_spx['cumsum_spx'] = (1+grid_growth_inflation_spx['spx']).cumprod()
    grid_growth_inflation_spx['cumsum_bt'] = (1+grid_growth_inflation_spx['bt_returns']).cumprod()

    ### DRAWDOWN ###
    def compute_drawdown(cumret):
        roll_max = cumret.cummax()
        drawdown = (cumret - roll_max) / roll_max
        return drawdown

    # Calculate drawdown series
    grid_growth_inflation_spx['drawdown_bt'] = compute_drawdown(grid_growth_inflation_spx['cumsum_bt'])
    grid_growth_inflation_spx['drawdown_spx'] = compute_drawdown(grid_growth_inflation_spx['cumsum_spx'])

    ### TABLE OF RESULTS ###
    grid_backtest_results = pd.DataFrame()
    grid_backtest_results['Strategy'] = ['GRID Model','SPX']
    grid_backtest_results['Mean Monthly Returns'] = [
        grid_growth_inflation_spx['bt_returns'].mean() * 100,
        grid_growth_inflation_spx['spx'].mean() * 100,
    ]
    grid_backtest_results['Ann. Returns'] = grid_backtest_results['Mean Monthly Returns'] * 12
    grid_backtest_results['Ann. Volatility'] = [
        (grid_growth_inflation_spx['bt_returns'].std() * 12**0.5) * 100,
        (grid_growth_inflation_spx['spx'].std() * 12**0.5) * 100,
    ]
    grid_backtest_results['Return/Risk'] = grid_backtest_results['Ann. Returns'] / grid_backtest_results['Ann. Volatility']

    ### PLOT ###
    fig = go.Figure()
    cols = ['cumsum_bt', 'cumsum_spx']
    labels = [
        'GRID',
        'SPX',
    ]
    colors = ['#5FB3FF', '#2DCDB2']
    for col, color, label in zip(cols, colors, labels):
        fig.add_trace(go.Scatter(x=grid_growth_inflation_spx.index, y=grid_growth_inflation_spx[col],
                                 mode='lines',
                                 name=label,
                                 line=dict(color=color)))
    fig.update_layout(
        title="GRID Z-Score Backtest",
        yaxis_title="Dollars",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    ### PLOT ###
    labels = ['GRID', 'SPX']
    draw_cols = ['drawdown_bt', 'drawdown_spx']
    line_colors = ['rgba(95,179,255,1)', 'rgba(45,205,178,1)']
    fill_colors = ['rgba(95,179,255,0.3)', 'rgba(45,205,178,0.3)']
    fig = go.Figure()
    for col, line, fill, label in zip(draw_cols, line_colors, fill_colors, labels):
        fig.add_trace(go.Scatter(
            x=grid_growth_inflation_spx.index,
            y=grid_growth_inflation_spx[col],
            mode='lines',
            name=label,
            line=dict(color=line, width=2),
            fill='tozeroy',
            fillcolor=fill,
            hovertemplate=f"{label}<br>Date: %{{x|%Y-%m-%d}}<br>Drawdown: %{{y:.2%}}<extra></extra>",
            showlegend=True
        ))
    fig.update_layout(
        title="Drawdown Analysis: GRID vs SPX",
        yaxis_title="Drawdown (%)",
        yaxis_tickformat='.0%',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor='bottom', y=1.02,
            xanchor='center', x=0.5,
            title=None
        ),
        margin=dict(l=40, r=40, t=70, b=40),
        plot_bgcolor='#f9f9f9'
    )
    st.plotly_chart(fig, use_container_width=True)

    ### TABLE ###
    st.title("GRID Backtest Results")
    cmap = LinearSegmentedColormap.from_list('red_white_green', ['#ff3333', '#ffffff', '#39b241'], N=256)
    styled = grid_backtest_results.style \
        .format({
        'Mean Monthly Returns': "{:.2f}%",
        'Ann. Returns': "{:.2f}%",
        'Ann. Volatility': "{:.2f}%",
        'Return/Risk': "{:.2f}",
    }) \
        .set_properties(
        subset=['Mean Monthly Returns', 'Ann. Returns', 'Ann. Volatility', 'Return/Risk'],
        **{'width': '500px'}
    ) \
        .background_gradient(cmap=cmap, subset=[
        'Mean Monthly Returns', 'Ann. Returns', 'Ann. Volatility', 'Return/Risk'
    ])

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write(styled, unsafe_allow_html=True)

def grid_pca_regime_performance(start, end, **kwargs):
    ### PCA ANALYSIS ###
    def rolling_pca(df, n_components=1, window=24):
        """Returns leading PC time series (length = original, nan at start) from rolling PCA on the columns of df."""
        pc_series = pd.Series(np.nan, index=df.index)
        for i in range(window - 1, len(df)):
            window_df = df.iloc[i - window + 1:i + 1]
            window_df = (window_df - window_df.mean()) / window_df.std()  # normalize
            pca = PCA(n_components=n_components)
            pc = pca.fit_transform(window_df.values)
            # Use last row's score for this time point
            pc_series.iloc[i] = pc[-1, 0]  # first principal component only
        return pc_series

    grid_growth_inflation_spx = pd.concat([
        rolling_pca(grid_growth_pct * grid_growth_corr_spx.dropna()),
        rolling_pca(grid_inflation_pct * grid_inflation_corr_spx.dropna()),
        spx_monthly_pct.shift(-1)
    ], axis=1).dropna()
    grid_growth_inflation_spx.columns = ['growth', 'inflation', 'spx']

    def regime_label(row):
        if row['inflation'] > 0 and row['growth'] > 0:
            return 0  # Reflation
        elif row['inflation'] > 0 and row['growth'] < 0:
            return 1  # Stagflation
        elif row['inflation'] < 0 and row['growth'] > 0:
            return 2  # Goldilocks
        elif row['inflation'] < 0 and row['growth'] < 0:
            return 3  # Deflation
        else:
            return np.nan

    df = grid_growth_inflation_spx.copy()
    df['regime_code'] = df.apply(regime_label, axis=1)
    df = df.dropna(subset=['regime_code']).copy()
    df['regime_code'] = df['regime_code'].astype(int)
    regime_labels = {
        0: 'Reflation',
        1: 'Stagflation',
        2: 'Goldilocks',
        3: 'Deflation'
    }
    regime_colors = {
        0: '#E74C3C',  # Reflation (red)
        1: '#F1C40F',  # Stagflation (yellow)
        2: '#27AE60',  # Goldilocks (green)
        3: '#2980B9'  # Deflation (blue)
    }
    df['regime_color'] = df['regime_code'].map(regime_colors)
    df['regime_label'] = df['regime_code'].map(regime_labels)

    reflation_regime = df[df['regime_label'] == 'Reflation']
    stagflation_regime = df[df['regime_label'] == 'Stagflation']
    goldilocks_regime = df[df['regime_label'] == 'Goldilocks']
    deflation_regime = df[df['regime_label'] == 'Deflation']

    reflation_mean_return = reflation_regime['spx'].mean() * 100
    stagflation_mean_return = stagflation_regime['spx'].mean() * 100
    goldilocks_mean_return = goldilocks_regime['spx'].mean() * 100
    deflation_mean_return = deflation_regime['spx'].mean() * 100

    reflation_ann_return = reflation_regime['spx'].mean() * 12 * 100
    stagflation_ann_return = stagflation_regime['spx'].mean() * 12 * 100
    goldilocks_ann_return = goldilocks_regime['spx'].mean() * 12 * 100
    deflation_ann_return = deflation_regime['spx'].mean() * 12 * 100

    reflation_volatility = reflation_regime['spx'].std() * (12 ** 0.5) * 100
    stagflation_volatility = stagflation_regime['spx'].std() * (12 ** 0.5) * 100
    goldilocks_volatility = goldilocks_regime['spx'].std() * (12 ** 0.5) * 100
    deflation_volatility = deflation_regime['spx'].std() * (12 ** 0.5) * 100

    reflation_win_ratio = reflation_regime[reflation_regime['spx'] > 0].shape[0] / reflation_regime.shape[0] * 100
    stagflation_win_ratio = stagflation_regime[stagflation_regime['spx'] > 0].shape[0] / stagflation_regime.shape[
        0] * 100
    goldilocks_win_ratio = goldilocks_regime[goldilocks_regime['spx'] > 0].shape[0] / goldilocks_regime.shape[0] * 100
    deflation_win_ratio = deflation_regime[deflation_regime['spx'] > 0].shape[0] / deflation_regime.shape[0] * 100

    total_rows = sum(
        [goldilocks_regime.shape[0], reflation_regime.shape[0], deflation_regime.shape[0], stagflation_regime.shape[0]])
    grid_results = pd.DataFrame()
    grid_results['Regime'] = [
        'Goldilocks (I-G+)',
        'Reflation (I+G+)',
        'Deflation (I-G-)',
        'Stagflation (I+G-)',
    ]
    grid_results['Mean Monthly Returns'] = [
        goldilocks_mean_return,
        reflation_mean_return,
        deflation_mean_return,
        stagflation_mean_return
    ]
    grid_results['Ann. Returns'] = [
        goldilocks_ann_return,
        reflation_ann_return,
        deflation_ann_return,
        stagflation_ann_return
    ]
    grid_results['Ann. Volatility'] = [
        goldilocks_volatility,
        reflation_volatility,
        deflation_volatility,
        stagflation_volatility
    ]
    grid_results['Return/Risk'] = grid_results['Ann. Returns'] / grid_results['Ann. Volatility']
    grid_results['Win Ratio'] = [
        goldilocks_win_ratio,
        reflation_win_ratio,
        deflation_win_ratio,
        stagflation_win_ratio
    ]
    grid_results['% of Occurrences'] = [
        goldilocks_regime.shape[0] / total_rows * 100,
        reflation_regime.shape[0] / total_rows * 100,
        deflation_regime.shape[0] / total_rows * 100,
        stagflation_regime.shape[0] / total_rows * 100
    ]

    st.title("GRID PCA Regime Performance")
    cmap = LinearSegmentedColormap.from_list('red_white_green', ['#ff3333', '#ffffff', '#39b241'], N=256)
    styled = grid_results.style \
        .format({
        'Mean Monthly Returns': "{:.2f}%",
        'Ann. Returns': "{:.2f}%",
        'Ann. Volatility': "{:.2f}%",
        'Return/Risk': "{:.2f}",
        'Win Ratio': "{:.2f}%",
        '% of Occurrences': "{:.2f}%"
    }) \
        .set_properties(
        subset=['Mean Monthly Returns', 'Ann. Returns', 'Ann. Volatility', 'Return/Risk', 'Win Ratio'],
        **{'width': '500px'}
    ) \
        .background_gradient(cmap=cmap, subset=[
        'Mean Monthly Returns', 'Ann. Returns', 'Ann. Volatility', 'Return/Risk', 'Win Ratio'
    ])

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write(styled, unsafe_allow_html=True)

    # --- Bonds Return Distribution ---
    st.title("GRID PCA Return Distributions")
    regimes = ['Reflation', 'Stagflation', 'Goldilocks', 'Deflation']
    regime_colors_plotly = {
        "Reflation": "#E74C3C",
        "Stagflation": "#F1C40F",
        "Goldilocks": "#27AE60",
        "Deflation": "#2980B9"
    }

    fig = make_subplots(rows=2, cols=2, subplot_titles=regimes)
    min_bound = df['spx'].min()
    max_bound = df['spx'].max()
    for i, regime in enumerate(regimes):
        row = i // 2 + 1
        col = i % 2 + 1
        subdata = df[df['regime_label'] == regime]
        fig.add_trace(
            go.Histogram(
                x=subdata['spx'].dropna(),
                name=regime,
                marker_color=regime_colors_plotly.get(regime, "#AAAAAA"),
                opacity=0.8,
                nbinsx=30
            ),
            row=row,
            col=col
        )
        fig.update_xaxes(title_text="Equity % Return", row=row, col=col, range=[min_bound, max_bound])
        fig.update_yaxes(title_text="Count", row=row, col=col)
    fig.update_layout(showlegend=False, height=600)
    st.plotly_chart(fig, use_container_width=True)

def grid_sector_performance(start, end, **kwargs):
    grid_growth_cross_mean_z = pd.DataFrame(grid_growth_z.mean(axis=1))
    grid_inflation_cross_mean_z = pd.DataFrame(grid_inflation_z.mean(axis=1))
    grid_growth_inflation_sectors = pd.concat([
        grid_growth_cross_mean_z,
        grid_inflation_cross_mean_z,
        spx_sectors_merge
    ], axis=1).dropna()

def grid_sector_backtest(start, end, **kwargs):
    grid_growth_cross_mean_z = pd.DataFrame(grid_growth_z.mean(axis=1))
    grid_inflation_cross_mean_z = pd.DataFrame(grid_inflation_z.mean(axis=1))
    grid_growth_inflation_sectors = pd.concat([
        grid_growth_cross_mean_z,
        grid_inflation_cross_mean_z,
        spx_sectors_merge
    ], axis=1).dropna()




