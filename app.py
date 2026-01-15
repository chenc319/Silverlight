### ---------------------------------------------------------------------------------------- ###
### -------------------------------- PACKAGES AND FUNCTIONS -------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

### IMPORT OTHER SCRIPTS ###
import streamlit as st
import pandas as pd
import functools as ft
import app_fund_flows
from archives import app_growth, app_inflation
import app_grid
import app_growth_inflation
import app_regime_summary
import app_barra
import app_liquidity
import app_sam_coreequity
import app_positioning
import app_cross_asset
import app_flowcluster
import app_yc
import app_momo
import app_td_combo

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
                "DeMark",
                "Monitors",
                "Analysis",
                'Passive Bubble'
                ]
    for section in sections:
        if section != current_section:
            st.session_state[f"{section}_selection"] = "Select an option..."

with st.sidebar:
    # Create a dictionary mapping sections to their options
    sections = {
        "Macro Regime Models": {
            "Select an option...": "Select an option...",
            "Regime Summary": "Regime Summary",
            "GRID Model": "GRID Model",
            "FlowCluster Model": "FlowCluster Model",
            "Cross-Asset Model": "Cross-Asset Model",
            "Prometheus Model": "Prometheus Model",
            "Yield Curve Model": "Yield Curve Model",
        },
        "DeMark": {
            "Select an option...": "Select an option...",
            "TD Combo": "TD Combo"
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
            "Barra Factor Model": "Barra Factor Model"
        },
        'Passive Bubble': {
            "Select an option...": "Select an option...",
            'Seasonality': 'Seasonality',
            'Momentum': 'Momentum',
            'Fund Flows': 'Fund Flows'
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

    st.markdown("### DeMark")
    demark = st.selectbox(
        "DeMark",
        list(sections["DeMark"].keys()),
        key="DeMark_selection",
        on_change=lambda: reset_other_selections("DeMark"),
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

    st.markdown("### Passive Bubble")
    passive_bubble_selection = st.selectbox(
        "Passive Bubble",
        list(sections["Passive Bubble"].keys()),
        key="Passive Bubble_selection",
        on_change=lambda: reset_other_selections("Passive Bubble"),
        label_visibility="collapsed"
    )

    # Set the current page based on any non-default selection
    page = "Select an option..."
    for selection in [macro_regime_models,
                      demark,
                      monitors_selection,
                      analysis_selection,
                      passive_bubble_selection
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
### ------------------------------------ REGIME SUMMARY ------------------------------------ ###
### ---------------------------------------------------------------------------------------- ###

elif page == 'Regime Summary':
    st.title('GRID')
    app_regime_summary.plot_grid_nowcast()
    st.title('FlowCluster')
    app_regime_summary.plot_flowcluster_nowcast()
    st.title('Cross-Asset')
    app_regime_summary.plot_crossasset_nowcast()
    st.title('Yield Curve')
    app_regime_summary.plot_yc_nowcast()

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
### --------------------------------- TAIL HEDGE PORTFOLIO --------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif page == 'Yield Curve Model':
    st.title('Yield Curve Regimes')
    app_yc.plot_colorcoded_regime()
    st.title('Equity Yield Curve Backtest')
    app_yc.equity_yc_results()

### ---------------------------------------------------------------------------------------- ###
### --------------------------------------- TD COMBO --------------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif page == 'TD Combo':
    st.title('GOOGL TD Combo Buy Case Study Example')
    app_td_combo.plot_googl_initial_results()

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

### ---------------------------------------------------------------------------------------- ###
### ------------------------------------- SEASONALITY -------------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif page == 'Seasonality':
    st.title('Equity Bond RV Spread')

### ---------------------------------------------------------------------------------------- ###
### -------------------------------------- FUND FLOWS -------------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif page == 'Fund Flows':
    st.title('Top 6 Fund Flows')
    app_fund_flows.display_etf_fund_flows()
    st.title('SPX Fund Flows Backtest')
    app_fund_flows.equity_fund_flow_results()
    st.title('MAGs Fund Flows Backtest')
    app_fund_flows.mags_fund_flow_results()

### ---------------------------------------------------------------------------------------- ###
### --------------------------------------- MOMENTUM --------------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

elif page == 'Momentum':
    st.title('Sector Rotation Backtest (excess vs. SPX)')
    app_momo.sector_x_spx_results()
    st.title('Sector Rotation Backtest (excess vs. BCOM)')
    app_momo.sector_x_bcom_results()
    st.title('Sector Rotation Backtest (excess vs. AGG)')
    app_momo.sector_x_bonds_results()
    st.title('Sector Rotation Backtest (excess vs. DXY)')
    app_momo.sector_x_dxy_results()
    st.title('Sector Rotation Backtest (excess vs. All Assets)')
    app_momo.sector_x_all_assets_results()
    st.title('Sector Rotation Backtest (excess vs. All Sectors)')
    app_momo.sector_x_all_sectors_results()

    st.title('MAGs Rotation Backtest (excess vs. SPX)')
    app_momo.mag7_x_spx_results()
    st.title('MAGs Rotation Backtest (excess vs. All MAGs)')
    app_momo.mag7_x_all_mags_results()




