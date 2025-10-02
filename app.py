### ---------------------------------------------------------------------------------------- ###
### -------------------------------- PACKAGES AND FUNCTIONS -------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

### IMPORT OTHER SCRIPTS ###
import streamlit as st
import pandas as pd
import functools as ft
import app_growth_inflation
import app_grid
import app_firv_regime
import app_barra_factors
import app_tail_hedge

import time

### FUNCTIONS ###
def merge_dfs(array_of_dfs):
    new_df = ft.reduce(lambda left,
                              right: pd.merge(left,
                                                    right,
                                                    left_index=True,
                                                    right_index=True,
                                                    how='outer'), array_of_dfs)
    return(new_df)

### ---------------------------------------------------------------------------------------- ###
### --------------------------------- CONFIGURE STREAMLIT ---------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

### CONFIGURE PAGE SETTINGS ###
st.set_page_config(
    page_title="Factor Models & Backtests",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .header-container {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        padding: 8px;
        background-color: white;
        z-index: 999;
        border-bottom: 1px solid #f0f2f6;
        font-size: 14px;
    }
    .main {
        margin-top: 60px;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 4px 8px;
        border-radius: 4px;
        display: inline-block;
        margin-right: 10px;
        font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)

### SIDEBAR ###
st.sidebar.title("Factor Models & Backtests")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('1999-12-31'))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('today'))

menu = st.sidebar.radio(
    "Go to section:",
    ['Growth & Inflation Model',
     'GRID Model',
     'Yield Curve Regimes',
     'Barra Factor Model',
     'Tail Hedge Portfolio']
)

### ---------------------------------------------------------------------------------------- ###
### --------------------------------- GROWTH AND INFLATION --------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

if menu == 'Growth & Inflation Model':
    app_growth_inflation.plot_growth_inflation(start_date, end_date)
    app_growth_inflation.plot_spx_sector_regimes(start_date, end_date)

### ---------------------------------------------------------------------------------------- ###
### --------------------------------- GROWTH AND INFLATION --------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif menu == 'GRID Model':
    app_grid.plot_grid_factors(start_date, end_date)
    app_grid.plot_grid_factors_regime_performance(start_date, end_date)
    st.title('GRID Model Backtest')
    app_grid.grid_z_score_backtest(start_date, end_date)
    

### ---------------------------------------------------------------------------------------- ###
### ------------------------------- YIELD CURVE REGIME MODEL ------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif menu == 'Yield Curve Regimes':
    st.title("Yield Curve Tenors by Regime")
    app_firv_regime.plot_treasury_yield_curves(start_date, end_date)


### ---------------------------------------------------------------------------------------- ###
### ---------------------------------- BARRA FACTOR MODEL ---------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif menu == 'Barra Factor Model':
    st.title("Barra Factors")
    app_barra_factors.plot_barra_factors(start_date,end_date)
    st.title("Z-Scored Factors: Average SPX Daily Return")
    app_barra_factors.plot_barra_factors(start_date,end_date)


### ---------------------------------------------------------------------------------------- ###
### --------------------------------- TAIL HEDGE PORTFOLIO --------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif menu == 'Tail Hedge Portfolio':
    st.title('Realized and Implied Volatility')
    app_tail_hedge.plot_veqtor_vix(start_date,end_date)


