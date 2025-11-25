### ---------------------------------------------------------------------------------------- ###
### -------------------------------- PACKAGES AND FUNCTIONS -------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

### IMPORT OTHER SCRIPTS ###
import streamlit as st
import pandas as pd
import functools as ft
import app_growth
import app_grid
import app_growth_inflation
import app_firv_regime
import app_barra
import app_inflation
import app_liquidity
import app_sam_coreequity
import app_positioning
import app_cross_asset
import app_flowcluster

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

def reset_other_selections(current_section):
    sections = ["Macro Regime Models",
                "Monitors",
                "Analysis"
                ]
    for section in sections:
        if section != current_section:
            st.session_state[f"{section}_selection"] = "Select an option..."

with st.sidebar:
    # Create a dictionary mapping sections to their options
    sections = {
        "Macro Regime Models": {
            "Select an option...": "Select an option...",
            "GRID Model": "GRID Model",
            "FlowCluster Model": "FlowCluster Model",
            "Cross-Asset Model": "Cross-Asset Model",
            "Prometheus Model": "Prometheus Model",
        },
        "Monitors": {
            "Select an option...": "Select an option...",
            "Growth Monitor": "Growth Monitor",
            "Inflation Monitor": "Inflation Monitor",
            "Liquidity Monitor": "Liquidity Monitor",
            "Positioning Monitor": "Positioning Monitor",
        },
        "Analysis": {
            "Select an option...": "Select an option...",
            "Growth & Inflation Study": "Growth & Inflation Study",
            "SAM Core Equity": "SAM Core Equity",
            "Yield Curve Regimes": "Yield Curve Regimes",
            "Barra Factor Model": "Barra Factor Model"
        }
    }

    # Initialize session state for each section if not exists
    for section in sections:
        if f"{section}_selection" not in st.session_state:
            st.session_state[f"{section}_selection"] = "Select an option..."

    # Create section headers and selectboxes
    st.markdown("### Macro Regime Models")
    macro_regime_models = st.selectbox(
        "Macro Regime Models",
        list(sections["Macro Regime Models"].keys()),
        key="Macro Regime Models_selection",
        on_change=lambda: reset_other_selections("Macro Regime Models"),
        label_visibility="collapsed"
    )

    st.markdown("### Monitors")
    monitors_selection = st.selectbox(
        "Monitors",
        list(sections["Monitors"].keys()),
        key="Monitors_selection",
        on_change=lambda: reset_other_selections("Monitors"),
        label_visibility="collapsed"
    )

    st.markdown("### Analysis")
    analysis_selection = st.selectbox(
        "Analysis",
        list(sections["Analysis"].keys()),
        key="Analysis_selection",
        on_change=lambda: reset_other_selections("Analysis"),
        label_visibility="collapsed"
    )

    # Set the current page based on any non-default selection
    page = "Select an option..."
    for selection in [macro_regime_models,
                      monitors_selection,
                      analysis_selection,
                      ]:
        if selection != "Select an option...":
            page = selection
            break

### ---------------------------------------------------------------------------------------- ###
### ----------------------------------- SAM CORE EQUITY ------------------------------------ ###
### ---------------------------------------------------------------------------------------- ###

if page == 'SAM Core Equity':
    st.title('Historical Performance')
    app_sam_coreequity.core_equity_mags_spx()
    st.title('Rolling Alpha')
    app_sam_coreequity.sam_core_equity_rolling_alpha()
    st.title('SAM Core Equity + MAGS Portfolios')
    app_sam_coreequity.core_equity_mag_backtest_simulation()
    st.title('Daily SAM CE vs. SPX')
    app_sam_coreequity.mock_daily_sam_ce_portfolio()

### ---------------------------------------------------------------------------------------- ###
### ----------------------------------- SAM CORE EQUITY ------------------------------------ ###
### ---------------------------------------------------------------------------------------- ###

elif page == 'Growth & Inflation Study':
    app_growth_inflation.plot_growth_inflation()
    app_growth_inflation.plot_spx_sectors_and_factors_regimes()

### ---------------------------------------------------------------------------------------- ###
### --------------------------------- GROWTH AND INFLATION --------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif page == 'GRID Model':
    st.title('Upcoming GRID Regime')
    app_grid.grid_regime_nowcast()
    st.title('GRID Equities Backtest')
    app_grid.grid_equity_backtest()
    st.title('GRID Bonds Backtest')
    app_grid.grid_bonds_backtest()
    st.title('GRID MAGS Backtest')
    app_grid.grid_mags_backtest()

### ---------------------------------------------------------------------------------------- ###
### ----------------------------------- FLOWCLUSTER MODEL ---------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif page == 'FlowCluster Model':
    st.title('Upcoming FlowCluster Regime')
    app_flowcluster.plot_colorcoded_regime()
    st.title('Equity Backtest')
    app_flowcluster.equity_flowcluster_results()
    st.title('Bonds Backtest')
    app_flowcluster.bonds_flowcluster_results()

### ---------------------------------------------------------------------------------------- ###
### ----------------------------------- PROMETHEUS MODEL ----------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif page == 'Cross-Asset Model':
    st.title('Cross-Asset Macro Regimes')
    app_cross_asset.plot_colorcoded_regime()
    st.title('Equity Backtest')
    app_cross_asset.equity_prometheus_results()
    st.title('Bonds Backtest')
    app_cross_asset.bonds_prometheus_results()
    st.title('BCOM Backtest')
    app_cross_asset.bcom_prometheus_results()


### ---------------------------------------------------------------------------------------- ###
### ---------------------------------------- GROWTH ---------------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif page == 'Growth Monitor':
    app_growth.plot_growth_predictor()
    app_growth.plot_growth_nowcast()

### ---------------------------------------------------------------------------------------- ###
### --------------------------------------- INFLATION -------------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif page == 'Inflation Monitor':
    app_inflation.plot_inflation_predictor()
    st.title('Inflation Nowcast')
    app_inflation.plot_cpi_nowcast()

### ---------------------------------------------------------------------------------------- ###
### ------------------------------- YIELD CURVE REGIME MODEL ------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif page == 'Yield Curve Regimes':
    st.title("Yield Curve Tenors by Regime")
    app_firv_regime.plot_treasury_yield_curves(start_date, end_date)

### ---------------------------------------------------------------------------------------- ###
### ---------------------------------- BARRA FACTOR MODEL ---------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif page == 'Barra Factor Model':
    st.title("Barra Factors")
    app_barra.plot_barra_factors(start_date,end_date)
    st.title("Barra Factor Prediction")
    app_barra.plot_barra_predictor()

### ---------------------------------------------------------------------------------------- ###
### --------------------------------- TAIL HEDGE PORTFOLIO --------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif page == 'SPX Monitor':
    st.title('Underlying Signals')
    app_positioning.plot_positioning_data()
    st.title('SPX OW/UW Backtest')
    app_positioning.plot_equity_pos_backtest()
    st.title('Bonds OW/UW Backtest')
    app_positioning.plot_bonds_pos_backtest()
    st.title('MAGS OW/UW Backtest')
    app_positioning.plot_mags_pos_backtest()

### ---------------------------------------------------------------------------------------- ###
### --------------------------------- TAIL HEDGE PORTFOLIO --------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif page == 'Liquidity Monitor':
    st.title('Fed Plumbing')
    app_liquidity.plot_fed_plumbing()
    st.title('Equity Fed Plumbing Backtest')
    app_liquidity.plot_equity_fed_plumbing_backtest()
    st.title('Equity Repo Venue Backtest')
    app_liquidity.plot_equity_repo_venue_backtest()




