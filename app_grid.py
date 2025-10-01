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
with open(Path(DATA_DIR) / 'hedge_eye_growth_variables.pkl', 'rb') as file:
    hedge_eye_growth_variables = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'hedge_eye_inflation_variables.pkl', 'rb') as file:
    hedge_eye_inflation_variables = pd.read_pickle(file)
hedge_eye_growth_pct = hedge_eye_growth_variables.pct_change().dropna()
hedge_eye_growth_pct.columns = growth_dict.keys()
hedge_eye_inflation_pct = hedge_eye_inflation_variables.pct_change().dropna()
hedge_eye_inflation_pct.columns = inflation_dict.keys()

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
    df = hedge_eye_growth_pct.copy().resample('ME').last()
    columns_to_plot = hedge_eye_growth_pct.columns
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
    df = hedge_eye_inflation_pct.copy().resample('ME').last()
    columns_to_plot = hedge_eye_inflation_pct.columns
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
    hedge_eye_inflation_pct

def plot_grid_factors_with_spx(start, end, **kwargs):
    spx_monthly_pct
