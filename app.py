### ---------------------------------------------------------------------------------------- ###
### -------------------------------- PACKAGES AND FUNCTIONS -------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

### IMPORT OTHER SCRIPTS ###
import streamlit as st
import pandas as pd
import functools as ft
import app_growth
import app_grid
import app_firv_regime
import app_barra
import app_inflation
import app_tail_hedge
import app_sam_coreequity

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
    page_title="SAM Research",
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
        margin-right: 10px;a
    }
    </style>
""", unsafe_allow_html=True)

### SIDEBAR ###
st.sidebar.title("SAM Research")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('1999-12-31'))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('today'))

menu = st.sidebar.radio(
    "Go to section:",
    ['SAM Core Equity',
     'GRID Model',
     'Growth Predictor',
     'Inflation Predictor',
     'Yield Curve Regimes',
     'Barra Factor Model',
     'Equity Positioning',
     'Tail Hedge Portfolio']
)

### ---------------------------------------------------------------------------------------- ###
### ----------------------------------- SAM CORE EQUITY ------------------------------------ ###
### ---------------------------------------------------------------------------------------- ###

if menu == 'SAM Core Equity':
    st.title('Historical Performance')
    app_sam_coreequity.core_equity_mags_spx()
    st.title('Rolling Alpha')
    app_sam_coreequity.sam_core_equity_rolling_alpha()
    st.title('SAM Core Equity + MAGS Portfolios')
    app_sam_coreequity.core_equity_mag_backtest_simulation()

### ---------------------------------------------------------------------------------------- ###
### --------------------------------- GROWTH AND INFLATION --------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif menu == 'GRID Model':
    st.title('Upcoming GRID Regime')
    app_grid.grid_regime_nowcast()
    st.title('GRID Equities Backtest')
    app_grid.grid_equity_backtest()
    st.title('GRID Bonds Backtest')
    app_grid.grid_bonds_backtest()

### ---------------------------------------------------------------------------------------- ###
### ---------------------------------------- GROWTH ---------------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif menu == 'Growth Predictor':
    app_growth.plot_growth_predictor()
    app_growth.plot_growth_nowcast()

### ---------------------------------------------------------------------------------------- ###
### ---------------------------------------- INFLATION ---------------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif menu == 'Inflation Predictor':
    app_inflation.plot_inflation_predictor()
    st.title('Inflation Nowcast')
    app_inflation.plot_cpi_nowcast()

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
    app_barra.plot_barra_factors(start_date,end_date)
    st.title("Barra Factor Prediction")
    app_barra.plot_barra_predictor()

### ---------------------------------------------------------------------------------------- ###
### --------------------------------- TAIL HEDGE PORTFOLIO --------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif menu == 'Tail Hedge Portfolio':
    st.title('Realized and Implied Volatility')
    app_tail_hedge.plot_veqtor_vix(start_date,end_date)


