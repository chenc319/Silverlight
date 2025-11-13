### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------------- LIQUIDITY ------------------------------------------------ ###
### ---------------------------------------------------------------------------------------------------------- ###
import streamlit

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
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

### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------------- LIQUIDITY ------------------------------------------------ ###
### ---------------------------------------------------------------------------------------------------------- ###

### SPX DATA ###
with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
    sp500 = pd.read_csv(file)
sp500.index = pd.to_datetime(sp500['Date']).values
sp500.drop('Date', axis=1, inplace=True)
spx_weekly = pd.DataFrame(sp500['Close']).resample('W-FRI').last()
spx_weekly.columns = ['spx']
spx_monthly = pd.DataFrame(sp500['Close']).resample('ME').last()
spx_monthly.columns = ['spx']

### BONDS DATA ###
with open(Path(DATA_DIR) / 'AGG.csv', 'rb') as file:
    bonds = pd.read_csv(file)
bonds.index = pd.to_datetime(bonds['Date']).values
bonds_weekly = pd.DataFrame(bonds['Close']).resample('W-FRI').last()
bonds_monthly = pd.DataFrame(bonds['Close']).resample('ME').last()

with open(Path(DATA_DIR) / 'mock_mags_monthly_pct.pkl', 'rb') as file:
    mock_mags_monthly_pct = pd.read_pickle(file)

### LIQUIDITY ###
with open(Path(DATA_DIR) / 'treasury.pkl', 'rb') as file:
    treasury = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'reserves.pkl', 'rb') as file:
    reserves = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'tga.pkl', 'rb') as file:
    tga = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'rrp_volume.pkl', 'rb') as file:
    rrp_volume = pd.read_pickle(file)

### REPO VENUES ###
with open(Path(DATA_DIR) / 'tri_volume_df.pkl', 'rb') as file:
    tri_volume_df = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'gcf_volume_df.pkl', 'rb') as file:
    gcf_volume_df = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'dvp_volume_df.pkl', 'rb') as file:
    dvp_volume_df = pd.read_pickle(file)

liquidity_df = merge_dfs([treasury,reserves,tga,rrp_volume]).dropna()
liquidity_df.index = liquidity_df.index.values
liquidity_df.columns = ['treasury','reserves','tga','onrrp']
liquidity_df = liquidity_df.resample('ME').last()

repo_venues_df = merge_dfs([tri_volume_df,gcf_volume_df,dvp_volume_df]).dropna()
repo_venues_df.index = repo_venues_df.index.values
repo_venues_df.columns = ['tri','gcf','dvp']
repo_venues_df = repo_venues_df.resample('ME').last()

### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------------- LIQUIDITY ------------------------------------------------ ###
### ---------------------------------------------------------------------------------------------------------- ###








### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------------- LIQUIDITY ------------------------------------------------ ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_fed_plumbing():
    streamlit_plot(df=liquidity_df,
                   columns_array=['treasury','reserves','tga','onrrp'],
                   colors_array=['#f8b62d', '#f8772d', '#2f90c5', '#67cbe7'],
                   graph_title='Liquidity Signals',
                   y_axis_label='Dollars')
    streamlit_plot(df=repo_venues_df,
                   columns_array=['tri','gcf','dvp'],
                   colors_array=['#f8b62d', '#f8772d', '#2f90c5'],
                   graph_title='Repo Venues',
                   y_axis_label='Dollars')
    streamlit_plot(df=repo_venues_df.diff(1),
                   columns_array=['tri', 'gcf', 'dvp'],
                   colors_array=['#f8b62d', '#f8772d', '#2f90c5'],
                   graph_title='Repo Venues 1st ROC',
                   y_axis_label='Dollars')
    streamlit_plot(df=repo_venues_df.diff(1).diff(1),
                   columns_array=['tri', 'gcf', 'dvp'],
                   colors_array=['#f8b62d', '#f8772d', '#2f90c5'],
                   graph_title='Repo Venues 2nd ROC',
                   y_axis_label='Dollars')

