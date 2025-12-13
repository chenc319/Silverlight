### ------------------------------------------------------------------------------------ ###
### --------------------------------------- MOMO --------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
from matplotlib import pyplot as plt
DATA_DIR = os.getenv('DATA_DIR', 'data')

### ------------------------------------------------------------------------------------ ###
### --------------------------------------- MOMO --------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

spx_sectors = {
    "SPLRCL": "Comm Services",
    "SPLRCD": "Cons Disc",
    "SPLRCS": "Cons Stap",
    "SPNY": "Energy",
    "SPSY": "Financial",
    "SPXHC": "Healthcare",
    "SPLRCI": "Industrial",
    "SPLRCM": "Materials",
    "SPLRCREC": "Real Estate",
    "SPLRCT": "Tech",
    "SPLRCU": "Utilities"
}

with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
    sp500 = pd.read_csv(file)
sp500.index = pd.to_datetime(sp500['Date']).values
sp500.drop('Date', axis=1, inplace=True)
spx_monthly = pd.DataFrame(sp500['Close']).resample('ME').last()
spx_monthly.columns = ['spx']

with open(Path(DATA_DIR) / 'AGG.csv', 'rb') as file:
    agg = pd.read_csv(file)
agg.index = pd.to_datetime(agg['Date']).values
agg.drop('Date', axis=1, inplace=True)
bonds_monthly = pd.DataFrame(agg['Close']).resample('ME').last()
bonds_monthly.columns = ['bonds']

with open(Path(DATA_DIR) / '^BCOM.csv', 'rb') as file:
    bcom = pd.read_csv(file)
bcom.index = pd.to_datetime(bcom['Date']).values
bcom.drop('Date', axis=1, inplace=True)
bcom_monthly = pd.DataFrame(bcom['Close']).resample('ME').last()
bcom_monthly.columns = ['bcom']

with open(Path(DATA_DIR) / 'DXY.csv', 'rb') as file:
    dxy = pd.read_csv(file)
dxy.index = pd.to_datetime(dxy['Date']).values
dxy.drop('Date', axis=1, inplace=True)
dxy_monthly = pd.DataFrame(dxy['Close']).resample('ME').last()
dxy_monthly.columns = ['dxy']

### SECTOR DATA ###
with open(Path(DATA_DIR) / ('sector_industry_returns.xlsx'), 'rb') as file:
    sector_df = pd.read_excel(file,sheet_name = 'sector')
sector_df.index = pd.to_datetime(sector_df['Dates']).values
sector_df.sort_index(inplace=True)
sector_df.drop('Dates', axis=1, inplace=True)
sector_df = sector_df.dropna()

### ------------------------------------------------------------------------------------ ###
### --------------------------------------- MOMO --------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

### CROSS ASSET ###
sectors_12_pct = sector_df.resample('ME').last().pct_change(12)
sectors_lag_pct = sector_df.resample('ME').last().pct_change(1).shift(-1)

sector_backtest = pd.DataFrame(columns = ['top4','mid4','bottom4','bt_returns'])
for row in sectors_12_pct.index:
    subset = sectors_12_pct.loc[row].sort_values(ascending=False)
    top4 = subset.index[:4]
    mid3 = subset.index[4:7]
    bottom3 = subset.index[7:11]
    future_top4 = sectors_lag_pct.loc[row,top4].mean() * 0.5
    future_mid3 = sectors_lag_pct.loc[row,mid3].mean() * 0.35
    future_bottom4 = sectors_lag_pct.loc[row,bottom3].mean() * 0.15
    total_bt_returns = future_top4 + future_mid3 + future_bottom4

    sector_backtest.loc[row,'top4'] = future_top4
    sector_backtest.loc[row,'mid4'] = future_mid3
    sector_backtest.loc[row,'bottom4'] = future_bottom4
    sector_backtest.loc[row,'bt_returns'] = total_bt_returns

sector_backtest['sector_benchmark'] = sectors_lag_pct.mean(axis=1)

sector_momo_return_metrics = return_metrics(
    sector_backtest[['bt_returns','sector_benchmark']],
    sector_backtest[['sector_benchmark']],
    12
)
sector_momo_return_metrics['Return/Risk']

### VOLATILITY SCALING ###
sector_realized_pct = sector_df.resample('ME').last().pct_change(1)
sector_rolling_vol = sector_realized_pct.rolling(3).std() * (12**0.5)
sector_vol_scale_weights = pd.DataFrame(columns = sector_realized_pct.columns)
for row in sector_rolling_vol.index:
    vol_today = sector_rolling_vol.loc[row]
    vol_today = vol_today.replace(0, np.nan)  # avoid div by zero
    inv_vol = 1.0 / vol_today
    inv_vol = inv_vol / inv_vol.sum()  # normalize to 1
    sector_vol_scale_weights.loc[row] = inv_vol

sector_vol_scale_bt = pd.DataFrame((sector_vol_scale_weights * sectors_lag_pct).dropna().sum(axis=1))
sector_vol_scale_bt.columns = ['bt_returns']
sector_vol_scale_bt['sector_benchmark'] = sectors_lag_pct.mean(axis=1)

sector_vol_scale_return_metrics = return_metrics(
    sector_vol_scale_bt[['bt_returns','sector_benchmark']],
    sector_vol_scale_bt[['sector_benchmark']],
    12
)
sector_vol_scale_return_metrics['Return/Risk']

### CROSS SECTIONAL SECTOR ROTATION MOMENTUM VS SPX ###
spx_12_pct = pd.DataFrame(sp500.resample('ME').last().pct_change(12)['Close'])
sector_x_ret = pd.DataFrame()
for each_sector in sectors_12_pct.columns:
    merge_df = merge_dfs([sectors_12_pct[each_sector],
                          spx_12_pct,
                          sectors_lag_pct[each_sector]]).dropna()
    merge_df.columns = ['sector','spx','lag_sector']
    merge_df['excess_return'] = merge_df['sector'] - merge_df['spx']
    sector_x_ret[each_sector] = merge_df['excess_return']

sector_excess_ret_mean = sector_x_ret.rolling(12).mean()
sector_excess_ret_std = sector_x_ret.rolling(12).std()
sector_excess_ret_z = (sector_x_ret - sector_excess_ret_mean) / sector_excess_ret_std

sector_excess_return_backtest = pd.DataFrame(columns = ['top4','mid4','bottom4','bt_returns'])
for row in sector_excess_ret_z.index:
    subset = sectors_12_pct.loc[row].sort_values(ascending=False)
    top4 = subset.index[:4]
    mid3 = subset.index[4:7]
    bottom3 = subset.index[7:11]
    future_top4 = sectors_lag_pct.loc[row,top4].mean() * 0.5
    future_mid3 = sectors_lag_pct.loc[row,mid3].mean() * 0.35
    future_bottom4 = sectors_lag_pct.loc[row,bottom3].mean() * 0.15
    total_bt_returns = future_top4 + future_mid3 + future_bottom4

    sector_excess_return_backtest.loc[row, 'top4'] = future_top4
    sector_excess_return_backtest.loc[row, 'mid4'] = future_mid3
    sector_excess_return_backtest.loc[row, 'bottom4'] = future_bottom4
    sector_excess_return_backtest.loc[row, 'bt_returns'] = total_bt_returns

sector_excess_return_backtest['sector_benchmark'] = sectors_lag_pct.mean(axis=1)

sector_cross_section_return_metrics = return_metrics(
    sector_excess_return_backtest[['bt_returns','sector_benchmark']],
    sector_excess_return_backtest[['sector_benchmark']],
    12
)
sector_cross_section_return_metrics['Return/Risk']

### ------------------------------------------------------------------------------------ ###
### --------------------------------------- MOMO --------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

def sector_momo_results():
    streamlit_plot(df=(1 + sector_backtest[['bt_returns',
                                            'sector_benchmark']]
                       ).cumprod() - 1,
                   columns_array=['bt_returns', 'sector_benchmark'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='Equities Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(sector_momo_return_metrics)

def sector_vol_scale_results():
    streamlit_plot(df=(1 + sector_vol_scale_bt[['bt_returns',
                                            'sector_benchmark']]
                       ).cumprod() - 1,
                   columns_array=['bt_returns', 'sector_benchmark'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='Equities Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(sector_vol_scale_return_metrics)

def sector_excess_return_results():
    streamlit_plot(df=(1 + sector_excess_return_backtest[['bt_returns',
                                            'sector_benchmark']]
                       ).cumprod() - 1,
                   columns_array=['bt_returns', 'sector_benchmark'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='Equities Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(sector_cross_section_return_metrics)

