### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------------- CFTC -------------------------------------------------- ###
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
### -------------------------------------------------- CFTC -------------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

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

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------------- CFTC -------------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### CALCULATE MONTHLY DIFFERENCES ###
spx_positioning_diff = spx_positioning_df.resample('ME').mean().dropna()
spx_positioning_diff.columns = ['spx_dealer','spx_asset_mgr','spx_lev_funds']
spx_positioning_diff['spx_total'] = spx_positioning_diff.sum(axis=1)
spx_positioning_diff['spx_lev_funds_1st_diff'] = spx_positioning_diff['spx_lev_funds'].diff(1)
spx_positioning_diff['spx_lev_funds_2nd_diff'] = spx_positioning_diff['spx_lev_funds_1st_diff'].diff(1)
spx_positioning_diff['spx_dealer_1st_diff'] = spx_positioning_diff['spx_dealer'].diff(1)
spx_positioning_diff['spx_dealer_2nd_diff'] = spx_positioning_diff['spx_dealer_1st_diff'].diff(1)
spx_positioning_diff = spx_positioning_diff.dropna()

emini_spx_positioning_diff = emini_spx_positioning_df.resample('ME').mean().dropna()
emini_spx_positioning_diff.columns = ['emini_dealer','emini_asset_mgr','emini_lev_funds']
emini_spx_positioning_diff['emini_total'] = emini_spx_positioning_diff.sum(axis=1)
emini_spx_positioning_diff['emini_lev_funds_1st_diff'] = emini_spx_positioning_diff['emini_lev_funds'].diff(1)
emini_spx_positioning_diff['emini_lev_funds_2nd_diff'] = emini_spx_positioning_diff['emini_lev_funds_1st_diff'].diff(1)
emini_spx_positioning_diff = emini_spx_positioning_diff.dropna()

vix_positioning_diff = vix_positioning_df.resample('ME').mean().dropna()
vix_positioning_diff.columns = ['vix_dealer','vix_asset_mgr','vix_lev_funds']
vix_positioning_diff['vix_total'] = vix_positioning_diff.sum(axis=1)
vix_positioning_diff['vix_lev_funds_1st_diff'] = vix_positioning_diff['vix_lev_funds'].diff(1)
vix_positioning_diff['vix_lev_funds_2nd_diff'] = vix_positioning_diff['vix_lev_funds_1st_diff'].diff(1)
vix_spx_positioning_diff = vix_positioning_diff.dropna()

### PREPARE WEEKLY CHANGES ###
spx_weekly_pct = spx_weekly.pct_change().dropna()
spx_monthly_pct = spx_monthly.pct_change().dropna()
bonds_weekly_pct = bonds_weekly.pct_change().dropna()
bonds_monthly_pct = bonds_monthly.pct_change().dropna()
positioning_merge_diff = merge_dfs([spx_positioning_diff, emini_spx_positioning_diff, vix_positioning_diff])

### MERGE DATA ###
spx_spx_cftc = merge_dfs([spx_positioning_diff,spx_monthly_pct.shift(-1)]).dropna()
bonds_spx_cftc = merge_dfs([spx_positioning_diff,bonds_monthly_pct.shift(-1)]).dropna()
mags_spx_cftc = merge_dfs([spx_positioning_diff,mock_mags_monthly_pct.shift(-1)]).dropna()

spx_emini_cftc = merge_dfs([emini_spx_positioning_diff,spx_monthly_pct.shift(-1)]).dropna()
bonds_emini_cftc = merge_dfs([emini_spx_positioning_diff,bonds_monthly_pct.shift(-1)]).dropna()
mags_emini_cftc = merge_dfs([emini_spx_positioning_diff,mock_mags_monthly_pct.shift(-1)]).dropna()

spx_vix_cftc = merge_dfs([vix_spx_positioning_diff,spx_monthly_pct.shift(-1)]).dropna()
bonds_vix_cftc = merge_dfs([vix_spx_positioning_diff,bonds_monthly_pct.shift(-1)]).dropna()
mags_vix_cftc = merge_dfs([vix_spx_positioning_diff,mock_mags_monthly_pct.shift(-1)]).dropna()

### ANALYTICS ###
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

spx_spx_cftc.columns
spx_cftc_bucket = bucket_signal_means(spx_spx_cftc,
                                        'spx_lev_funds_1st_diff',
                                        'spx_lev_funds_2nd_diff',
                                        'spx')
bonds_cftc_bucket = bucket_signal_means(bonds_spx_cftc,
                                        'spx_lev_funds_1st_diff',
                                        'spx_lev_funds_2nd_diff',
                                        'Close')
mags_cftc_bucket = bucket_signal_means(mags_spx_cftc,
                                        'spx_lev_funds_1st_diff',
                                        'spx_lev_funds_2nd_diff',
                                        'mags')

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------------- CFTC -------------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### FUNCTIONS FOR BACKTEST ###
def ow_uw_positioning_backtest(row,signal_colname):
    if row[signal_colname] < -1:
        return 1
    elif row[signal_colname] > -1 and row[signal_colname] < 0:
        return 0.75
    elif row[signal_colname] > 0 and row[signal_colname] < 1:
        return 0.5
    elif row[signal_colname] > 1:
        return 0.25

def ow_uw_positioning_backtest(row,signal_colname_1,signal_colname_2):
    if row[signal_colname_1] > 0 and row[signal_colname_2] > 0:
        return 0.75
    elif row[signal_colname_1] > 0 and row[signal_colname_2] <0:
        return 0.5
    elif row[signal_colname_1] < 0 and row[signal_colname_2] > 0:
        return 0.25
    elif row[signal_colname_1] < 0 and row[signal_colname_2] < 0:
        return 1

### BACKTESTS ###
spx_spx_cftc['weights'] = spx_spx_cftc.apply(
    lambda row: ow_uw_positioning_backtest(row, 'spx_lev_funds_1st_diff','spx_lev_funds_2nd_diff'), axis=1
)
bonds_spx_cftc['weights'] = bonds_spx_cftc.apply(
    lambda row: ow_uw_positioning_backtest(row, 'spx_lev_funds_1st_diff','spx_lev_funds_2nd_diff'), axis=1
)
mags_spx_cftc['weights'] = mags_spx_cftc.apply(
    lambda row: ow_uw_positioning_backtest(row, 'spx_lev_funds_1st_diff','spx_lev_funds_2nd_diff'), axis=1
)
spx_spx_cftc['bt_returns'] = (spx_spx_cftc['weights'] * spx_spx_cftc['spx'])
bonds_spx_cftc['bt_returns'] = (bonds_spx_cftc['weights'] * bonds_spx_cftc['Close'])
mags_spx_cftc['bt_returns'] = (mags_spx_cftc['weights'] * mags_spx_cftc['mags'])

### RETURN METRICS ###
spx_return_metrics = return_metrics(
    spx_spx_cftc[['bt_returns','spx']],
    spx_spx_cftc[['spx']],
    12
)
bonds_return_metrics = return_metrics(
    bonds_spx_cftc[['bt_returns','Close']],
    bonds_spx_cftc[['Close']],
    12
)
mags_return_metrics = return_metrics(
    mags_spx_cftc[['bt_returns','mags']],
    mags_spx_cftc[['mags']],
    12
)

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------------- CFTC -------------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_positioning_data():
    streamlit_plot(df=spx_positioning_diff,
                   columns_array=spx_positioning_diff.columns,
                   colors_array=["#7393B3", "#57666A", "#5F9EA0", "#4682B4", "#6082B6"],
                   graph_title='Weekly Changes in SPX Futures',
                   y_axis_label='%')
    streamlit_plot(df=emini_spx_positioning_diff,
                   columns_array=emini_spx_positioning_diff.columns,
                   colors_array=["#7393B3", "#57666A", "#5F9EA0", "#4682B4", "#6082B6"],
                   graph_title='Weekly Changes in E-Mini Futures',
                   y_axis_label='%')
    streamlit_plot(df=vix_positioning_diff,
                   columns_array=vix_positioning_diff.columns,
                   colors_array=["#7393B3", "#57666A", "#5F9EA0", "#4682B4", "#6082B6"],
                   graph_title='Weekly Changes in VIX Futures',
                   y_axis_label='%')

def plot_equity_pos_backtest():
    streamlit_plot(df=(1 + spx_spx_cftc).cumprod() -1,
                   columns_array=['bt_returns','spx'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='Equities Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(spx_return_metrics)

def plot_bonds_pos_backtest():
    streamlit_plot(df=(1 + bonds_spx_cftc).cumprod() -1,
                   columns_array=['bt_returns','Close'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='Bonds Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(bonds_return_metrics)

def plot_mags_pos_backtest():
    streamlit_plot(df=(1 + mags_spx_cftc).cumprod() -1,
                   columns_array=['bt_returns','mags'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='MAGS Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(mags_return_metrics)
