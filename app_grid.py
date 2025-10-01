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
grid_growth_pct = grid_growth_variables.pct_change().dropna()
grid_growth_pct.columns = growth_dict.keys()
grid_inflation_pct = grid_inflation_variables.pct_change().dropna()
grid_inflation_pct.columns = inflation_dict.keys()

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
    df = grid_growth_variables.resample('ME').last()
    columns_to_plot = grid_growth_variables.columns
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

    ### PLOT ###
    st.title('GRID Inflation Factors')
    df = grid_inflation_pct.copy().resample('ME').last()
    columns_to_plot = grid_inflation_pct.columns
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

def plot_grid_factors_z_score_backtest(start, end, **kwargs):
    ### CALCULATION ###
    grid_growth_cross_mean_z = pd.DataFrame(grid_growth_z.mean(axis=1))
    grid_inflation_cross_mean_z = pd.DataFrame(grid_inflation_z.mean(axis=1))
    grid_growth_inflation_spx = merge_dfs([grid_growth_cross_mean_z,
                                           grid_inflation_cross_mean_z,
                                           spx_monthly_pct.shift(-1)]).dropna()
    grid_growth_inflation_spx.columns = ['growth','inflation','spx']

    ### REGIME FUNCTION ###
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

    ### REGIME CLASSIFICATION ###
    reflation_regime = grid_growth_inflation_spx[
        (grid_growth_inflation_spx['inflation'] > 0) &
        (grid_growth_inflation_spx['growth'] > 0)
    ]
    stagflation_regime = grid_growth_inflation_spx[
        (grid_growth_inflation_spx['inflation'] > 0) &
        (grid_growth_inflation_spx['growth'] < 0)
    ]
    goldilocks_regime = grid_growth_inflation_spx[
        (grid_growth_inflation_spx['inflation'] < 0) &
        (grid_growth_inflation_spx['growth'] > 0)
    ]
    deflation_regime = grid_growth_inflation_spx[
        (grid_growth_inflation_spx['inflation'] < 0) &
        (grid_growth_inflation_spx['growth'] < 0)
    ]
    grid_growth_inflation_spx['regime_code'] = grid_growth_inflation_spx.apply(regime_label, axis=1)

    grid_growth_inflation_spx[grid_growth_inflation_spx['spx']]

    positive_growth = grid_growth_inflation_spx[grid_growth_inflation_spx['growth']>0]
    len(positive_growth[positive_growth['spx']>0]) / len(positive_growth)
    positive_inflation = grid_growth_inflation_spx[grid_growth_inflation_spx['inflation']>0]
    len(positive_inflation[positive_inflation['spx']>0]) / len(positive_inflation)

    len(grid_growth_inflation_spx[grid_growth_inflation_spx['spx']>0]) / len(grid_growth_inflation_spx)
    (grid_growth_inflation_spx['spx'].mean() * 12) / (grid_growth_inflation_spx['spx'].std() * 12**0.5)
    plt.plot(grid_growth_inflation_spx['spx'])
    plt.show()

    regime_colors = {
        0: '#E74C3C',  # Reflation (red)
        1: '#F1C40F',  # Stagflation (yellow)
        2: '#27AE60',  # Goldilocks (green)
        3: '#2980B9'  # Deflation (blue)
    }

    df = grid_growth_inflation_spx.copy()
    df = df.dropna(subset=['regime_code']).copy()
    df = df[~df.index.duplicated(keep='first')]
    df['regime_code'] = df['regime_code'].astype(int)
    df['regime_color'] = df['regime_code'].map(regime_colors)

    reflation_mean_return = reflation_regime[['spx']].mean(axis=0) * 100
    stagflation_mean_return = stagflation_regime[['spx']].mean(axis=0) * 100
    goldilocks_mean_return = goldilocks_regime[['spx']].mean(axis=0) * 100
    deflation_mean_return = deflation_regime[['spx']].mean(axis=0) * 100

    reflation_ann_return = reflation_regime[['spx']].mean(axis=0) * 12 * 100
    stagflation_ann_return = stagflation_regime[['spx']].mean(axis=0) * 12 * 100
    goldilocks_ann_return = goldilocks_regime[['spx']].mean(axis=0) * 12 * 100
    deflation_ann_return = deflation_regime[['spx']].mean(axis=0) * 12 * 100

    reflation_volatility = reflation_regime[['spx']].std(axis=0) * (12**0.5) * 100
    stagflation_volatility = stagflation_regime[['spx']].std(axis=0) * (12**0.5) * 100
    goldilocks_volatility = goldilocks_regime[['spx']].std(axis=0) * (12**0.5) * 100
    deflation_volatility = deflation_regime[['spx']].std(axis=0) * (12**0.5) * 100

    reflation_win_ratio = len(reflation_regime[reflation_regime['spx']>0]) / len(reflation_regime) * 100
    stagflation_win_ratio = len(stagflation_regime[stagflation_regime['spx']>0]) / len(stagflation_regime) * 100
    goldilocks_win_ratio = len(goldilocks_regime[goldilocks_regime['spx']>0]) / len(goldilocks_regime) * 100
    deflation_win_ratio = len(deflation_regime[deflation_regime['spx']>0]) / len(deflation_regime) * 100

    grid_results = pd.DataFrame()
    grid_results['Regime'] = [
        'Goldilocks (I-G+)',
        'Reflation (I+G+)',
        'Deflation (I-G-)',
        'Stagflation (I+G-)',
    ]
    grid_results['Mean Monthly Returns'] = [goldilocks_mean_return[0],
                                            reflation_mean_return[0],
                                            deflation_mean_return[0],
                                            stagflation_mean_return[0]]
    grid_results['Ann. Returns'] = [goldilocks_ann_return[0],
                                    reflation_ann_return[0],
                                    deflation_ann_return[0],
                                    stagflation_ann_return[0]]
    grid_results['Ann. Volatility'] = [goldilocks_volatility[0],
                                       reflation_volatility[0],
                                       deflation_volatility[0],
                                       stagflation_volatility[0]]
    grid_results['Return/Risk'] = grid_results['Ann. Returns'] / grid_results['Ann. Volatility']
    grid_results['Win Ratio'] = [goldilocks_win_ratio,
                                 reflation_win_ratio,
                                 deflation_win_ratio,
                                 stagflation_win_ratio]
    grid_results['Win Ratio'] = round(grid_results['Win Ratio'])
    total_rows = len(goldilocks_regime) + len(reflation_regime) + len(deflation_regime) + len(stagflation_regime)
    grid_results['% of Occurrences'] = [
        len(goldilocks_regime) / total_rows,
        len(reflation_regime) / total_rows,
        len(deflation_regime) / total_rows,
        len(stagflation_regime) / total_rows
    ]
    grid_results['% of Occurrences'] = round((grid_results['% of Occurrences'] * 100))


def grid_z_score_corr_backtest(start, end, **kwargs):
    grid_growth_mean = grid_growth_pct.rolling(12).mean()
    grid_growth_std = grid_growth_pct.rolling(12).std()
    grid_growth_z = (grid_growth_pct - grid_growth_mean) / grid_growth_std
    grid_growth_z.columns = growth_dict.values()
    grid_growth_z_spx_merge = merge_dfs([grid_growth_z,spx_monthly_pct])

    reference_growth_col = grid_growth_z_spx_merge.columns[-1]
    grid_growth_rolling_corr = pd.DataFrame({
        col: grid_growth_z_spx_merge[col].rolling(12).corr(grid_growth_z_spx_merge[reference_growth_col])
        for col in grid_growth_z_spx_merge.columns[:-1]
    })



