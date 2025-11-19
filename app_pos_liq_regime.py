### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------- POSITIONING LIQUIDITY REGIME -------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
DATA_DIR = os.getenv('DATA_DIR', 'data')

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------- POSITIONING LIQUIDITY REGIME -------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### SPX DATA ###
with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
    sp500 = pd.read_csv(file)
sp500.index = pd.to_datetime(sp500['Date']).values
sp500.drop('Date', axis=1, inplace=True)
spx_weekly = pd.DataFrame(sp500['Close']).resample('W-FRI').last()
spx_weekly.columns = ['spx']
spx_daily = pd.DataFrame(sp500['Close'])
spx_daily.columns = ['spx']
spx_monthly = pd.DataFrame(sp500['Close']).resample('ME').last()
spx_monthly.columns = ['spx']

### BONDS DATA ###
with open(Path(DATA_DIR) / 'AGG.csv', 'rb') as file:
    bonds = pd.read_csv(file)
bonds.index = pd.to_datetime(bonds['Date']).values
bonds_daily = pd.DataFrame(bonds['Close'])
bonds_daily.columns = ['bonds']
bonds_weekly = pd.DataFrame(bonds['Close']).resample('W-FRI').last()
bonds_weekly.columns = ['bonds']
bonds_monthly = pd.DataFrame(bonds['Close']).resample('ME').last()
bonds_monthly.columns = ['bonds']

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

### POSITIONING DATA ###
with open(Path(DATA_DIR) / 'spx_positioning_df.pkl', 'rb') as file:
    spx_positioning_df = pd.read_pickle(file)[['dealer_spread','asset_mgr_spread','lev_funds_spread']]
    spx_positioning_df.index = pd.to_datetime(spx_positioning_df.index)
with open(Path(DATA_DIR) / 'emini_spx_positioning_df.pkl', 'rb') as file:
    emini_spx_positioning_df = pd.read_pickle(file)[['dealer_spread','asset_mgr_spread','lev_funds_spread']]
    emini_spx_positioning_df.index = pd.to_datetime(emini_spx_positioning_df.index)
with open(Path(DATA_DIR) / 'vix_positioning_df.pkl', 'rb') as file:
    vix_positioning_df = pd.read_pickle(file)[['dealer_spread','asset_mgr_spread','lev_funds_spread']]
    vix_positioning_df.index = pd.to_datetime(vix_positioning_df.index)

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------- POSITIONING LIQUIDITY REGIME -------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### AGGREGATE DATA AND RESAMPLE ###
liquidity_df = merge_dfs([treasury,reserves,tga,rrp_volume]).dropna()
liquidity_df.index = liquidity_df.index.values
liquidity_df.columns = ['treasury','reserves','tga','onrrp']
liquidity_df = liquidity_df.resample('ME').last()

repo_venues_df = merge_dfs([tri_volume_df,gcf_volume_df,dvp_volume_df]).dropna()
repo_venues_df.index = repo_venues_df.index.values
repo_venues_df.columns = ['tri','gcf','dvp']
repo_venues_df = repo_venues_df.resample('ME').last()

### MERGE DFS ###
liquidity_spx_merge = merge_dfs([liquidity_df,spx_monthly.pct_change().shift(-1)])
liquidity_bonds_merge = merge_dfs([liquidity_df,bonds_monthly.pct_change().shift(-1)])
repo_venues_spx_merge = merge_dfs([repo_venues_df,spx_monthly.pct_change().shift(-1)])
repo_venues_bonds_merge = merge_dfs([repo_venues_df,bonds_monthly.pct_change().shift(-1)])

### CALCULATE MONTHLY DIFFERENCES ###
spx_positioning_diff = spx_positioning_df.resample('ME').mean().dropna()
spx_positioning_diff.columns = ['spx_dealer','spx_asset_mgr','spx_lev_funds']
spx_positioning_diff['spx_total'] = spx_positioning_diff.sum(axis=1)
spx_positioning_diff['spx_lev_funds_1st_diff'] = spx_positioning_diff['spx_lev_funds'].diff(1)
spx_positioning_diff['spx_lev_funds_2nd_diff'] = spx_positioning_diff['spx_lev_funds_1st_diff'].diff(1)
spx_positioning_diff['spx_dealer_1st_diff'] = spx_positioning_diff['spx_dealer'].diff(1)
spx_positioning_diff['spx_dealer_2nd_diff'] = spx_positioning_diff['spx_dealer_1st_diff'].diff(1)
spx_positioning_diff = spx_positioning_diff.dropna()

### MERGE DATA ###
spx_spx_cftc = merge_dfs([spx_positioning_diff,spx_monthly.pct_change().shift(-1)]).dropna()
bonds_spx_cftc = merge_dfs([spx_positioning_diff,bonds_monthly.pct_change().shift(-1)]).dropna()

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------- POSITIONING LIQUIDITY REGIME -------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###


