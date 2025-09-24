### ---------------------------------------------------------------------------------------- ###
### -------------------------------- PACKAGES AND FUNCTIONS -------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

### IMPORT OTHER SCRIPTS ###
import streamlit as st
import pandas as pd
import functools as ft

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
    page_title="Silverlight Regime Dashboard",
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
st.sidebar.title("Multifactor Regime Models")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2019-12-31'))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('today'))

menu = st.sidebar.radio(
    "Go to section:",
    ['Liquidity Stress',
     'Fed Balance Sheet',
     'Repo Activity',
     'Money Markets',
     'Primary Dealers',
     'Shadow Banks',
     'Treasury Auctions',
     'UST Positioning',
     'STIR Positioning',
     'TRACE Model']
)

### ---------------------------------------------------------------------------------------- ###
### -------------------------------------- RISK CHECKS ------------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

if menu == 'Liquidity Stress':
    st.title("SOFR Spreads")
    app_liquidity_stress.plot_sofr_iorb(start_date, end_date)
    st.title("Repo Spreads")
    app_liquidity_stress.plot_iorb_spreads(start_date, end_date)


