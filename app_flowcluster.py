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

spx_positioning_diff = spx_positioning_df.resample('ME').mean().dropna()
spx_positioning_diff.columns = ['spx_dealer','spx_asset_mgr','spx_lev_funds']

flow_cluster_df = merge_dfs([
    liquidity_df['treasury'].diff(),
    liquidity_df['treasury'].diff().diff(),
    spx_positioning_diff['spx_lev_funds'].diff(),
    spx_positioning_diff['spx_lev_funds'].diff().diff(),
    spx_monthly.pct_change().shift(-1),
    bonds_monthly.pct_change().shift(-1)
]).dropna()

flow_cluster_df.columns = [
    'liquidity_1_roc',
    'liquidity_2_roc',
    'positioning_1_roc',
    'positioning_2_roc',
    'spx',
    'bonds']

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------- POSITIONING LIQUIDITY REGIME -------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def regime_label(row):
    if row['liquidity_2_roc'] > 0 and row['positioning_2_roc'] < 0:
        return 'Goldilocks'
    elif row['liquidity_2_roc'] < 0 and row['positioning_2_roc'] < 0:
        return 'Reflation'
    elif row['liquidity_2_roc'] > 0 and row['positioning_2_roc'] > 0:
        return 'Stagflation'
    elif row['liquidity_2_roc'] < 0 and row['positioning_2_roc'] > 0:
        return 'Deflation'
    return np.nan

def ow_uw_flowcluster_backtest(row,signal_colname_1,signal_colname_2):
    if row[signal_colname_1] > 0 and row[signal_colname_2] < 0:
        return 1
    elif row[signal_colname_1] < 0 and row[signal_colname_2] < 0:
        return 0.9
    elif row[signal_colname_1] > 0 and row[signal_colname_2] > 0:
        return 0.8
    elif row[signal_colname_1] < 0 and row[signal_colname_2] > 0:
        return 0.7

flow_cluster_df['weights'] = flow_cluster_df.apply(
    lambda row: ow_uw_flowcluster_backtest(row,
                                           'liquidity_2_roc',
                                           'positioning_2_roc'), axis=1
)
flow_cluster_df['regime_label'] = flow_cluster_df.apply(regime_label, axis=1)
flow_cluster_df['equity_bt'] = flow_cluster_df['spx'] * flow_cluster_df['weights']
flow_cluster_df['bonds_bt'] = flow_cluster_df['bonds'] * flow_cluster_df['weights']
flow_cluster_df = flow_cluster_df.dropna()

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------- POSITIONING LIQUIDITY REGIME -------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------- flowcluster REGIME MODEL ----------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

equities_flowcluster_return_metrics = return_metrics(
    flow_cluster_df[['equity_bt','spx']],
    flow_cluster_df[['spx']],
    12
)
equities_flowcluster_return_metrics['Return/Risk']

bonds_flowcluster_return_metrics = return_metrics(
    flow_cluster_df[['bonds_bt','bonds']],
    flow_cluster_df[['bonds']],
    12
)
bonds_flowcluster_return_metrics['Return/Risk']

def plot_colorcoded_regime():
    regime_merge = merge_dfs([
        pd.DataFrame(flow_cluster_df['regime_label']),
        spx_monthly,
        pd.DataFrame(flow_cluster_df['equity_bt'])
    ]).dropna()
    color_coded_regime_plot(regime_merge,
                            y_col='spx',
                            regime_col='closest_regime',
                            title="S&P 500 Level by Macro Regime")

def equity_flowcluster_results():
    streamlit_plot(df=(1 + flow_cluster_df[['equity_bt', 'spx']]).cumprod() - 1,
                   columns_array=['equity_bt', 'spx'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='Equities Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(equities_flowcluster_return_metrics)

def bonds_flowcluster_results():
    streamlit_plot(df=(1 + flow_cluster_df[['bonds_bt', 'bonds']]).cumprod() - 1,
                   columns_array=['bonds_bt', 'bonds'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='Bonds Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(bonds_flowcluster_return_metrics)





