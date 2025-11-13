### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------------- LIQUIDITY ------------------------------------------------ ###
### ---------------------------------------------------------------------------------------------------------- ###

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

### MAGS ###
with open(Path(DATA_DIR) / 'mock_mags_daily_pct.pkl', 'rb') as file:
    mock_mags_daily_pct = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'mock_mags_weekly_pct.pkl', 'rb') as file:
    mock_mags_weekly_pct = pd.read_pickle(file)
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
repo_venues_spx_merge = merge_dfs([repo_venues_df,spx_monthly.pct_change().shift(-1)])

### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------------- LIQUIDITY ------------------------------------------------ ###
### ---------------------------------------------------------------------------------------------------------- ###

### MERGE DFS ###
liquidity_spx_merge['treasury_1st_roc'] = liquidity_spx_merge['treasury'].diff()
liquidity_spx_merge['treasury_2nd_roc'] = liquidity_spx_merge['treasury_1st_roc'].diff()
liquidity_spx_merge['reserves_1st_roc'] = liquidity_spx_merge['reserves'].diff()
liquidity_spx_merge['reserves_2nd_roc'] = liquidity_spx_merge['reserves_1st_roc'].diff()
liquidity_spx_merge['tga_1st_roc'] = liquidity_spx_merge['tga'].diff()
liquidity_spx_merge['tga_2nd_roc'] = liquidity_spx_merge['tga_1st_roc'].diff()
liquidity_spx_merge['onrrp_1st_roc'] = liquidity_spx_merge['onrrp'].diff()
liquidity_spx_merge['onrrp_2nd_roc'] = liquidity_spx_merge['onrrp_1st_roc'].diff()
liquidity_spx_merge = liquidity_spx_merge.dropna()

### MERGE DFS ###
repo_venues_spx_merge['tri_1st_roc'] = repo_venues_spx_merge['tri'].diff()
repo_venues_spx_merge['tri_2nd_roc'] = repo_venues_spx_merge['tri_1st_roc'].diff()
repo_venues_spx_merge['gcf_1st_roc'] = repo_venues_spx_merge['gcf'].diff()
repo_venues_spx_merge['gcf_2nd_roc'] = repo_venues_spx_merge['gcf_1st_roc'].diff()
repo_venues_spx_merge['dvp_1st_roc'] = repo_venues_spx_merge['dvp'].diff()
repo_venues_spx_merge['dvp_2nd_roc'] = repo_venues_spx_merge['dvp_1st_roc'].diff()
repo_venues_spx_merge = repo_venues_spx_merge.dropna()

def bucket_signal_means(df, signal1_col, signal2_col, returns_col):
    buckets = {
        'above_0_above_0': (df[signal1_col] > 0) & (df[signal2_col] > 0),
        'above_0_below_0': (df[signal1_col] > 0) & (df[signal2_col] <= 0),
        'below_0_above_0': (df[signal1_col] <= 0) & (df[signal2_col] > 0),
        'below_0_below_0': (df[signal1_col] <= 0) & (df[signal2_col] <= 0)
    }
    means = {}
    for name, mask in buckets.items():
        means[name] = df.loc[mask, returns_col].mean()
    return means

bucket_signal_means(liquidity_spx_merge,
                    'treasury_1st_roc',
                    'treasury_2nd_roc',
                    'spx')
bucket_signal_means(liquidity_spx_merge,
                    'reserves_1st_roc',
                    'reserves_2nd_roc',
                    'spx')
bucket_signal_means(liquidity_spx_merge,
                    'tga_1st_roc',
                    'tga_2nd_roc',
                    'spx')
bucket_signal_means(liquidity_spx_merge,
                    'onrrp_1st_roc',
                    'onrrp_2nd_roc',
                    'spx')

bucket_signal_means(repo_venues_spx_merge,
                    'tri_1st_roc',
                    'tri_2nd_roc',
                    'spx')
bucket_signal_means(repo_venues_spx_merge,
                    'gcf_1st_roc',
                    'gcf_2nd_roc',
                    'spx')
bucket_signal_means(repo_venues_spx_merge,
                    'dvp_1st_roc',
                    'dvp_2nd_roc',
                    'spx')

### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------------- LIQUIDITY ------------------------------------------------ ###
### ---------------------------------------------------------------------------------------------------------- ###

### BACKTEST WEIGHTS FUNCTION ###
def ow_uw_fed_plumbing_backtest(row,signal_colname_1,signal_colname_2):
    if row[signal_colname_1] > 0 and row[signal_colname_2] > 0:
        return 1
    elif row[signal_colname_1] > 0 and row[signal_colname_2] < 0:
        return 0.50
    elif row[signal_colname_1] < 0 and row[signal_colname_2] > 0:
        return 0.75
    elif row[signal_colname_1] < 0 and row[signal_colname_2] < 0:
        return 0.25

def ow_uw_repo_venue_backtest(row,signal_colname_1,signal_colname_2):
    if row[signal_colname_1] > 0 and row[signal_colname_2] > 0:
        return 0.5
    elif row[signal_colname_1] > 0 and row[signal_colname_2] < 0:
        return 1
    elif row[signal_colname_1] < 0 and row[signal_colname_2] > 0:
        return 0.25
    elif row[signal_colname_1] < 0 and row[signal_colname_2] < 0:
        return 0.75

### BACKTESTS ###
liquidity_spx_merge['weights'] = liquidity_spx_merge.apply(
    lambda row: ow_uw_fed_plumbing_backtest(row,
                                           'treasury_1st_roc',
                                           'treasury_2nd_roc'), axis=1
)
liquidity_spx_merge['bt_returns'] = (liquidity_spx_merge['weights'] * liquidity_spx_merge['spx'])

repo_venues_spx_merge['weights'] = liquidity_spx_merge.apply(
    lambda row: ow_uw_repo_venue_backtest(row,
                                           'dvp_1st_roc',
                                           'dvp_2nd_roc'), axis=1
)
repo_venues_spx_merge['bt_returns'] = (repo_venues_spx_merge['weights'] * liquidity_spx_merge['spx'])

### RETURN METRICS ###
spx_fed_plumbing_return_metrics = return_metrics(
    liquidity_spx_merge[['bt_returns','spx']],
    liquidity_spx_merge[['spx']],
    12
)
spx_fed_plumbing_return_metrics['Return/Risk']

spx_repo_venues_return_metrics = return_metrics(
    repo_venues_spx_merge[['bt_returns','spx']],
    repo_venues_spx_merge[['spx']],
    12
)
spx_repo_venues_return_metrics['Return/Risk']

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

def plot_equity_fed_plumbing_backtest():
    streamlit_plot(df=(1 + liquidity_spx_merge).cumprod() -1,
                   columns_array=['bt_returns','spx'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='Equities Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(spx_fed_plumbing_return_metrics)

def plot_equity_repo_venue_backtest():
    streamlit_plot(df=(1 + repo_venues_spx_merge).cumprod() - 1,
                   columns_array=['bt_returns', 'spx'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='Equities Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(spx_repo_venues_return_metrics)


