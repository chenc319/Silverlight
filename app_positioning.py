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

### BONDS DATA ###
with open(Path(DATA_DIR) / 'AGG.csv', 'rb') as file:
    bonds_weekly = pd.read_csv(file)
bonds_weekly.index = pd.to_datetime(bonds_weekly['Date']).values
bonds_weekly = pd.DataFrame(bonds_weekly['Close']).resample('W-FRI').last()

### MAGS DATA ###
mags_tickers = ['GOOGL','AMZN','AAPL','META','MSFT','NVDA','TSLA']
each_mags_df = pd.DataFrame()
for mag_ticker in mags_tickers:
    mag_string = mag_ticker + '.csv'
    with open(Path(DATA_DIR) / mag_string, 'rb') as file:
        mag_df = pd.read_csv(file)
        mag_df.index = pd.to_datetime(mag_df['Date'].values)
        close_df = pd.DataFrame(mag_df['Close'])
        close_df.columns = [mag_ticker]
    each_mags_df = merge_dfs([each_mags_df, close_df])
each_mags_df = each_mags_df.dropna()
mags_weekly_pct = each_mags_df.resample('W-FRI').last().pct_change().dropna()

### MAGS WEIGHTS ###
with open(Path(DATA_DIR) / 'mags_weights.xlsx', 'rb') as file:
    mags_weights_df = pd.read_excel(file,sheet_name='Sheet1')
    mags_weights_df.index = mags_weights_df['Date'].values
    mags_weights_df.drop('Date', axis=1, inplace=True)
    mags_weights_df['sum'] = mags_weights_df.sum(axis=1)
normalized_mags_weights = pd.DataFrame(columns = ['GOOGL','AMZN','AAPL','META','MSFT','NVDA','TSLA'])
normalized_mags_pct = pd.DataFrame(columns = ['GOOGL','AMZN','AAPL','META','MSFT','NVDA','TSLA'])
for col in normalized_mags_weights.columns:
    normalized_mags_weights[col] = (mags_weights_df[col] / mags_weights_df['sum']).resample('W-FRI').last()
    col_df = merge_dfs([mags_weekly_pct[col],normalized_mags_weights[col]]).ffill().dropna()
    col_df.columns = ['pct','weights']
    normalized_mags_pct[col] = col_df['pct'] * col_df['weights']

mock_mags_weekly_pct = pd.DataFrame(normalized_mags_pct.sum(axis=1))
mock_mags_weekly_pct.columns = ['mags']

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------------- CFTC -------------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

spx_positioning_df = spx_positioning_df.resample('W-FRI').last().dropna().diff(1)
spx_positioning_df.columns = ['spx_dealer','spx_asset_mgr','spx_lev_funds']
emini_spx_positioning_df = emini_spx_positioning_df.resample('W-FRI').last().dropna().diff(1)
emini_spx_positioning_df.columns = ['emini_dealer','emini_asset_mgr','emini_lev_funds']
vix_positioning_df = vix_positioning_df.resample('W-FRI').last().dropna().diff(1)
vix_positioning_df.columns = ['vix_dealer','vix_asset_mgr','vix_lev_funds']

spx_weekly_pct = spx_weekly.pct_change().dropna()
bonds_weekly_pct = bonds_weekly.pct_change().dropna()
positioning_df = merge_dfs([spx_positioning_df, emini_spx_positioning_df, vix_positioning_df])
positioning_df['total'] = positioning_df.sum(axis=1)

rolling_cftc_mean = positioning_df.rolling(12).mean()
rolling_cftc_std = positioning_df.rolling(12).std()
rolling_cftc_z_score = ((positioning_df - rolling_cftc_mean) / rolling_cftc_std).dropna()

spx_spx_cftc_z = merge_dfs([rolling_cftc_z_score,spx_weekly_pct.shift(-1)]).dropna()
bonds_spx_cftc_z = merge_dfs([rolling_cftc_z_score,bonds_weekly_pct.shift(-1)]).dropna()
mags_spx_cftc_z = merge_dfs([rolling_cftc_z_score,mock_mags_weekly_pct.shift(-1)]).dropna()


### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------------- CFTC -------------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

return_col = 'spx'
df = spx_spx_cftc_z.copy()

def assets_cftc_description(df,return_col):
    z_cols = [col for col in df.columns if col != return_col]
    bins = [-np.inf, -1, 0, 1, np.inf]
    labels = [
        "< -1", "-1 to 0",
        "0 to 1", "> 1"
    ]
    result = pd.DataFrame(index=z_cols, columns=labels)
    for col in z_cols:
        binned = pd.cut(df[col], bins=bins, labels=labels)
        gp = df.groupby(binned)[return_col].mean()
        result.loc[col] = gp.reindex(labels).values

    return result

spx_cftc_bucket = assets_cftc_description(spx_spx_cftc_z,'spx')
bonds_cftc_bucket = assets_cftc_description(bonds_spx_cftc_z,'Close')
mags_cftc_bucket = assets_cftc_description(mags_spx_cftc_z,'mags')

spx_pos_bt =





