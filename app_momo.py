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
mags_monthly = each_mags_df.resample('ME').last()
mags_monthly_pct = mags_monthly.pct_change()

### MAGS WEIGHTS ###
with open(Path(DATA_DIR) / 'mags_weights.xlsx', 'rb') as file:
    mags_weights_df = pd.read_excel(file,sheet_name='Sheet1')
    mags_weights_df.index = mags_weights_df['Date'].values
    mags_weights_df.drop('Date', axis=1, inplace=True)
    mags_weights_df['sum'] = mags_weights_df.sum(axis=1)
normalized_mags_weights = pd.DataFrame(columns = ['GOOGL','AMZN','AAPL','META','MSFT','NVDA','TSLA'])
normalized_mags_pct = pd.DataFrame(columns = ['GOOGL','AMZN','AAPL','META','MSFT','NVDA','TSLA'])
for col in normalized_mags_weights.columns:
    normalized_mags_weights[col] = (mags_weights_df[col] / mags_weights_df['sum']).resample('ME').last()
    col_df = merge_dfs([mags_monthly_pct[col],normalized_mags_weights[col]]).ffill().dropna()
    col_df.columns = ['pct','weights']
    normalized_mags_pct[col] = col_df['pct'] * col_df['weights']

mock_mags_monthly_pct = pd.DataFrame(normalized_mags_pct.sum(axis=1))
mock_mags_monthly_pct.columns = ['mags']

### PREPARE DFS ###
sectors_12_pct = sector_df.resample('ME').last().pct_change(12)
sectors_lag_pct = sector_df.resample('ME').last().pct_change(1).shift(-1)
mag7_12_pct = mags_monthly.pct_change(12)
mag7_realized_pct = mags_monthly.pct_change()
mag7_lagged_pct = mags_monthly.pct_change().shift(-1)

### ------------------------------------------------------------------------------------ ###
### --------------------------------------- MOMO --------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

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

sector_x_spx_backtest = pd.DataFrame(columns = ['top4','mid4','bottom4','bt_returns'])
for row in sector_excess_ret_z.index:
    subset = sector_excess_ret_z.loc[row].sort_values(ascending=False)
    top4 = subset.index[:4]
    mid3 = subset.index[4:7]
    bottom3 = subset.index[7:11]
    future_top4 = sectors_lag_pct.loc[row,top4].mean() * 0.5
    future_mid3 = sectors_lag_pct.loc[row,mid3].mean() * 0.35
    future_bottom4 = sectors_lag_pct.loc[row,bottom3].mean() * 0.15
    total_bt_returns = future_top4 + future_mid3 + future_bottom4

    sector_x_spx_backtest.loc[row, 'top4'] = future_top4
    sector_x_spx_backtest.loc[row, 'mid4'] = future_mid3
    sector_x_spx_backtest.loc[row, 'bottom4'] = future_bottom4
    sector_x_spx_backtest.loc[row, 'bt_returns'] = total_bt_returns

sector_x_spx_backtest['sector_benchmark'] = sectors_lag_pct.mean(axis=1)

sector_x_spx_return_metrics = return_metrics(
    sector_x_spx_backtest[['bt_returns','sector_benchmark']],
    sector_x_spx_backtest[['sector_benchmark']],
    12
)
sector_x_spx_return_metrics['Return/Risk']

### ------------------------------------------------------------------------------------ ###
### --------------------------------------- MOMO --------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

### CROSS SECTIONAL SECTOR ROTATION MOMENTUM VS SPX ###
bcom_12_pct = pd.DataFrame(bcom.resample('ME').last().pct_change(12)['Close'])
sector_x_ret = pd.DataFrame()
for each_sector in sectors_12_pct.columns:
    merge_df = merge_dfs([sectors_12_pct[each_sector],
                          bcom_12_pct,
                          sectors_lag_pct[each_sector]]).dropna()
    merge_df.columns = ['sector','bcom','lag_sector']
    merge_df['excess_return'] = merge_df['sector'] - merge_df['bcom']
    sector_x_ret[each_sector] = merge_df['excess_return']

sector_excess_ret_mean = sector_x_ret.rolling(12).mean()
sector_excess_ret_std = sector_x_ret.rolling(12).std()
sector_excess_ret_z = (sector_x_ret - sector_excess_ret_mean) / sector_excess_ret_std

sector_x_bcom_backtest = pd.DataFrame(columns = ['top4','mid4','bottom4','bt_returns'])
for row in sector_excess_ret_z.index:
    subset = sector_excess_ret_z.loc[row].sort_values(ascending=False)
    top4 = subset.index[:4]
    mid3 = subset.index[4:7]
    bottom3 = subset.index[7:11]
    future_top4 = sectors_lag_pct.loc[row,top4].mean() * 0.5
    future_mid3 = sectors_lag_pct.loc[row,mid3].mean() * 0.35
    future_bottom4 = sectors_lag_pct.loc[row,bottom3].mean() * 0.15
    total_bt_returns = future_top4 + future_mid3 + future_bottom4

    sector_x_bcom_backtest.loc[row, 'top4'] = future_top4
    sector_x_bcom_backtest.loc[row, 'mid4'] = future_mid3
    sector_x_bcom_backtest.loc[row, 'bottom4'] = future_bottom4
    sector_x_bcom_backtest.loc[row, 'bt_returns'] = total_bt_returns

sector_x_bcom_backtest['sector_benchmark'] = sectors_lag_pct.mean(axis=1)

sector_x_bcom_return_metrics = return_metrics(
    sector_x_bcom_backtest[['bt_returns','sector_benchmark']],
    sector_x_bcom_backtest[['sector_benchmark']],
    12
)
sector_x_bcom_return_metrics['Return/Risk']

### ------------------------------------------------------------------------------------ ###
### --------------------------------------- MOMO --------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

### CROSS SECTIONAL SECTOR ROTATION MOMENTUM VS SPX ###
bonds_12_pct = pd.DataFrame(agg.resample('ME').last().pct_change(12)['Close'])
sector_x_ret = pd.DataFrame()
for each_sector in sectors_12_pct.columns:
    merge_df = merge_dfs([sectors_12_pct[each_sector],
                          bonds_12_pct,
                          sectors_lag_pct[each_sector]]).dropna()
    merge_df.columns = ['sector','bonds','lag_sector']
    merge_df['excess_return'] = merge_df['sector'] - merge_df['bonds']
    sector_x_ret[each_sector] = merge_df['excess_return']

sector_excess_ret_mean = sector_x_ret.rolling(12).mean()
sector_excess_ret_std = sector_x_ret.rolling(12).std()
sector_excess_ret_z = (sector_x_ret - sector_excess_ret_mean) / sector_excess_ret_std

sector_x_bonds_backtest = pd.DataFrame(columns = ['top4','mid4','bottom4','bt_returns'])
for row in sector_excess_ret_z.index:
    subset = sector_excess_ret_z.loc[row].sort_values(ascending=False)
    top4 = subset.index[:4]
    mid3 = subset.index[4:7]
    bottom3 = subset.index[7:11]
    future_top4 = sectors_lag_pct.loc[row,top4].mean() * 0.5
    future_mid3 = sectors_lag_pct.loc[row,mid3].mean() * 0.35
    future_bottom4 = sectors_lag_pct.loc[row,bottom3].mean() * 0.15
    total_bt_returns = future_top4 + future_mid3 + future_bottom4

    sector_x_bonds_backtest.loc[row, 'top4'] = future_top4
    sector_x_bonds_backtest.loc[row, 'mid4'] = future_mid3
    sector_x_bonds_backtest.loc[row, 'bottom4'] = future_bottom4
    sector_x_bonds_backtest.loc[row, 'bt_returns'] = total_bt_returns

sector_x_bonds_backtest['sector_benchmark'] = sectors_lag_pct.mean(axis=1)

sector_x_bonds_return_metrics = return_metrics(
    sector_x_bonds_backtest[['bt_returns','sector_benchmark']],
    sector_x_bonds_backtest[['sector_benchmark']],
    12
)
sector_x_bonds_return_metrics['Return/Risk']

### ------------------------------------------------------------------------------------ ###
### --------------------------------------- MOMO --------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

### CROSS SECTIONAL SECTOR ROTATION MOMENTUM VS SPX ###
dxy_12_pct = pd.DataFrame(dxy.resample('ME').last().pct_change(12)['Close'])
sector_x_ret = pd.DataFrame()
for each_sector in sectors_12_pct.columns:
    merge_df = merge_dfs([sectors_12_pct[each_sector],
                          dxy_12_pct,
                          sectors_lag_pct[each_sector]]).dropna()
    merge_df.columns = ['sector','dxy','lag_sector']
    merge_df['excess_return'] = merge_df['sector'] - merge_df['dxy']
    sector_x_ret[each_sector] = merge_df['excess_return']

sector_excess_ret_mean = sector_x_ret.rolling(12).mean()
sector_excess_ret_std = sector_x_ret.rolling(12).std()
sector_excess_ret_z = (sector_x_ret - sector_excess_ret_mean) / sector_excess_ret_std

sector_x_dxy_backtest = pd.DataFrame(columns = ['top4','mid4','bottom4','bt_returns'])
for row in sector_excess_ret_z.index:
    subset = sector_excess_ret_z.loc[row].sort_values(ascending=False)
    top4 = subset.index[:4]
    mid3 = subset.index[4:7]
    bottom3 = subset.index[7:11]
    future_top4 = sectors_lag_pct.loc[row,top4].mean() * 0.5
    future_mid3 = sectors_lag_pct.loc[row,mid3].mean() * 0.35
    future_bottom4 = sectors_lag_pct.loc[row,bottom3].mean() * 0.15
    total_bt_returns = future_top4 + future_mid3 + future_bottom4

    sector_x_dxy_backtest.loc[row, 'top4'] = future_top4
    sector_x_dxy_backtest.loc[row, 'mid4'] = future_mid3
    sector_x_dxy_backtest.loc[row, 'bottom4'] = future_bottom4
    sector_x_dxy_backtest.loc[row, 'bt_returns'] = total_bt_returns

sector_x_dxy_backtest['sector_benchmark'] = sectors_lag_pct.mean(axis=1)

sector_x_dxy_return_metrics = return_metrics(
    sector_x_dxy_backtest[['bt_returns','sector_benchmark']],
    sector_x_dxy_backtest[['sector_benchmark']],
    12
)
sector_x_dxy_return_metrics['Return/Risk']

### ------------------------------------------------------------------------------------ ###
### --------------------------------------- MOMO --------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

### MOMENTUM RELATIVE TO ALL OF THE CROSS ASSETS ###
cross_asset_pct = merge_dfs([
    spx_monthly.rename(columns={'spx': 'spx'}),
    bonds_monthly.rename(columns={'bonds': 'bonds'}),
    bcom_monthly.rename(columns={'bcom': 'bcom'}),
    dxy_monthly.rename(columns={'dxy': 'dxy'})
]).pct_change(12).dropna()

sector_cross_asset_z = pd.DataFrame()
for each_sector in sectors_12_pct.columns:
    merge_df = merge_dfs([sectors_12_pct[each_sector],
                          cross_asset_pct]).dropna()
    merge_df['x_spx'] = merge_df[each_sector] - merge_df['spx']
    merge_df['x_bonds'] = merge_df[each_sector] - merge_df['bonds']
    merge_df['x_bcom'] = merge_df[each_sector] - merge_df['bcom']
    merge_df['x_dxy'] = merge_df[each_sector] - merge_df['dxy']

    merge_df_spx_mean = merge_df['x_spx'].rolling(12).mean()
    merge_df_spx_std = merge_df['x_spx'].rolling(12).std()
    merge_df['x_spx_z'] =( merge_df['x_spx'] - merge_df_spx_mean) / merge_df_spx_std

    merge_df_bonds_mean = merge_df['x_bonds'].rolling(12).mean()
    merge_df_bonds_std = merge_df['x_bonds'].rolling(12).std()
    merge_df['x_bonds_z'] = (merge_df['x_bonds'] - merge_df_bonds_mean) / merge_df_bonds_std

    merge_df_bcom_mean = merge_df['x_bcom'].rolling(12).mean()
    merge_df_bcom_std = merge_df['x_bcom'].rolling(12).std()
    merge_df['x_bcom_z'] = (merge_df['x_bcom'] - merge_df_bcom_mean) / merge_df_bcom_std

    merge_df_dxy_mean = merge_df['x_dxy'].rolling(12).mean()
    merge_df_dxy_std = merge_df['x_dxy'].rolling(12).std()
    merge_df['x_dxy_z'] = (merge_df['x_dxy'] - merge_df_dxy_mean) / merge_df_dxy_std

    merge_df['rolling_corr_sign_spx'] = rolling_beta_sign(
        merge_df[each_sector], merge_df['spx'], 6, thresh=-0.2
    )
    merge_df['rolling_corr_sign_bonds'] = rolling_beta_sign(
        merge_df[each_sector], merge_df['bonds'], 6, thresh=-0.2
    )
    merge_df['rolling_corr_sign_bcom'] = rolling_beta_sign(
        merge_df[each_sector], merge_df['bcom'], 6, thresh=-0.2
    )
    merge_df['rolling_corr_sign_dxy'] = rolling_beta_sign(
        merge_df[each_sector], merge_df['dxy'], 6, thresh=-0.2
    )

    merge_df['x_cross_asset_z'] = (
        (merge_df['x_spx_z'] * 0.25 * merge_df['rolling_corr_sign_spx']) +
        (merge_df['x_bonds_z'] * 0.25 * merge_df['rolling_corr_sign_bonds']) +
        (merge_df['x_bcom_z'] * 0.25 * merge_df['rolling_corr_sign_bcom']) +
        (merge_df['x_dxy_z'] * 0.25 * merge_df['rolling_corr_sign_dxy'])
    )
    sector_cross_asset_z[each_sector] = merge_df['x_cross_asset_z']

sector_x_all_assets_backtest = pd.DataFrame(columns = ['top4','mid4','bottom4','bt_returns'])
for row in sector_cross_asset_z.index[:-1]:
    subset = sector_cross_asset_z.loc[row].sort_values(ascending=False)
    top4 = subset.index[:4]
    mid3 = subset.index[4:7]
    bottom3 = subset.index[7:11]
    future_top4 = sectors_lag_pct.loc[row,top4].mean() * 0.5
    future_mid3 = sectors_lag_pct.loc[row,mid3].mean() * 0.35
    future_bottom4 = sectors_lag_pct.loc[row,bottom3].mean() * 0.15
    total_bt_returns = future_top4 + future_mid3 + future_bottom4

    sector_x_all_assets_backtest.loc[row, 'top4'] = future_top4
    sector_x_all_assets_backtest.loc[row, 'mid4'] = future_mid3
    sector_x_all_assets_backtest.loc[row, 'bottom4'] = future_bottom4
    sector_x_all_assets_backtest.loc[row, 'bt_returns'] = total_bt_returns

sector_x_all_assets_backtest['sector_benchmark'] = sectors_lag_pct.mean(axis=1)

sector_x_all_assets_return_metrics = return_metrics(
    sector_x_all_assets_backtest[['bt_returns','sector_benchmark']],
    sector_x_all_assets_backtest[['sector_benchmark']],
    12
)
sector_x_all_assets_return_metrics['Return/Risk']

### ------------------------------------------------------------------------------------ ###
### --------------------------------------- MOMO --------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

### MOMENTUM RELATIVE TO ALL OF THE SECTORS ###
sector_cross_asset_z = pd.DataFrame()
for each_sector in sectors_12_pct.columns:
    df_subset = sectors_12_pct.drop(columns=[each_sector])
    merge_df = merge_dfs([df_subset,
                          sectors_12_pct[each_sector]]).dropna()
    comp_sector_excess_return = pd.DataFrame()
    comp_sector_rolling_beta = pd.DataFrame()
    comp_sector_excess_return_beta_sign = pd.DataFrame()
    for comp_sector in df_subset.columns:
        comp_sector_excess_return[comp_sector] = (
                sectors_12_pct[each_sector] - sectors_12_pct[comp_sector])
        comp_sector_rolling_beta[comp_sector] = rolling_beta_sign(
            sectors_12_pct[each_sector],
            sectors_12_pct[comp_sector],
            12,
            0
        )
        comp_sector_excess_return_beta_sign[comp_sector] = (
            comp_sector_excess_return[comp_sector] *
            comp_sector_rolling_beta[comp_sector]
        )
    comp_sector_mean = comp_sector_excess_return_beta_sign.rolling(12).mean()
    comp_sector_std = comp_sector_excess_return_beta_sign.rolling(12).std()
    comp_sector_z = ((comp_sector_excess_return_beta_sign -
                      comp_sector_mean) / comp_sector_std).mean(axis=1)
    sector_cross_asset_z[each_sector] = comp_sector_z
sector_cross_asset_z = sector_cross_asset_z.dropna()


sector_x_all_assets_backtest = pd.DataFrame(columns = ['top4','mid4','bottom4','bt_returns'])
for row in sector_cross_asset_z.index[:-1]:
    subset = sector_cross_asset_z.loc[row].sort_values(ascending=False)
    top4 = subset.index[:4]
    mid3 = subset.index[4:7]
    bottom3 = subset.index[7:11]
    future_top4 = sectors_lag_pct.loc[row,top4].mean() * 0.5
    future_mid3 = sectors_lag_pct.loc[row,mid3].mean() * 0.35
    future_bottom4 = sectors_lag_pct.loc[row,bottom3].mean() * 0.15
    total_bt_returns = future_top4 + future_mid3 + future_bottom4

    sector_x_all_assets_backtest.loc[row, 'top4'] = future_top4
    sector_x_all_assets_backtest.loc[row, 'mid4'] = future_mid3
    sector_x_all_assets_backtest.loc[row, 'bottom4'] = future_bottom4
    sector_x_all_assets_backtest.loc[row, 'bt_returns'] = total_bt_returns

sector_x_all_assets_backtest['sector_benchmark'] = sectors_lag_pct.mean(axis=1)

sector_x_all_assets_return_metrics = return_metrics(
    sector_x_all_assets_backtest[['bt_returns','sector_benchmark']],
    sector_x_all_assets_backtest[['sector_benchmark']],
    12
)
sector_x_all_assets_return_metrics['Return/Risk']

### ------------------------------------------------------------------------------------ ###
### --------------------------------------- MOMO --------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

### MAG7 MOMENTUM RELATIVE TO SPX ###
mags_x_ret = pd.DataFrame()
for each_mag in mag7_12_pct.columns:
    merge_df = merge_dfs([mag7_12_pct[each_mag],
                          spx_12_pct,
                          mag7_lagged_pct[each_mag]]).dropna()
    merge_df.columns = ['mag','spx','lag_mag']
    merge_df['excess_return'] = merge_df['mag'] - merge_df['spx']
    mags_x_ret[each_mag] = merge_df['excess_return']

mags_excess_ret_mean = mags_x_ret.rolling(12).mean()
mags_excess_ret_std = mags_x_ret.rolling(12).std()
mags_excess_ret_z = (mags_x_ret - mags_excess_ret_mean) / mags_excess_ret_std

mags_x_spx_backtest = pd.DataFrame(columns = ['top1','mid1','mid2','bottom1','bt_returns'])
for row in mags_excess_ret_z.index:
    subset = mags_excess_ret_z.loc[row].sort_values(ascending=False)
    top1 = subset.index[:2]
    mid1 = subset.index[2:4]
    mid2 = subset.index[4:6]
    bottom1 = subset.index[6:7]
    future_top1 = mag7_lagged_pct.loc[row,top1].mean() * 0.4
    future_mid1 = mag7_lagged_pct.loc[row,mid1].mean() * 0.3
    future_mid2 = mag7_lagged_pct.loc[row,mid2].mean() * 0.2
    future_bottom1 = mag7_lagged_pct.loc[row,bottom1].mean() * 0.1
    total_bt_returns = future_top1 + future_mid1 + future_mid2 + future_bottom1

    mags_x_spx_backtest.loc[row, 'top1'] = future_top1
    mags_x_spx_backtest.loc[row, 'mid1'] = future_mid1
    mags_x_spx_backtest.loc[row, 'mid2'] = future_mid2
    mags_x_spx_backtest.loc[row, 'bottom1'] = future_bottom1
    mags_x_spx_backtest.loc[row, 'bt_returns'] = total_bt_returns

mags_x_spx_backtest['mags_benchmark'] = mag7_lagged_pct.mean(axis=1)

mags_x_spx_return_metrics = return_metrics(
    mags_x_spx_backtest[['bt_returns','mags_benchmark']],
    mags_x_spx_backtest[['mags_benchmark']],
    12
)
mags_x_spx_return_metrics['Return/Risk']

### ------------------------------------------------------------------------------------ ###
### --------------------------------------- MOMO --------------------------------------- ###
### ------------------------------------------------------------------------------------ ###

def sector_x_spx_results():
    streamlit_plot(df=(1 + sector_x_spx_backtest[['bt_returns',
                                            'sector_benchmark']]
                       ).cumprod() - 1,
                   columns_array=['bt_returns', 'sector_benchmark'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='Equities Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(sector_x_spx_return_metrics)

def sector_x_bcom_results():
    streamlit_plot(df=(1 + sector_x_bcom_backtest[['bt_returns',
                                            'sector_benchmark']]
                       ).cumprod() - 1,
                   columns_array=['bt_returns', 'sector_benchmark'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='Equities Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(sector_x_bcom_return_metrics)

def sector_x_bonds_results():
    streamlit_plot(df=(1 + sector_x_bonds_backtest[['bt_returns',
                                            'sector_benchmark']]
                       ).cumprod() - 1,
                   columns_array=['bt_returns', 'sector_benchmark'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='Equities Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(sector_x_bonds_return_metrics)

def sector_x_dxy_results():
    streamlit_plot(df=(1 + sector_x_dxy_backtest[['bt_returns',
                                            'sector_benchmark']]
                       ).cumprod() - 1,
                   columns_array=['bt_returns', 'sector_benchmark'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='Equities Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(sector_x_dxy_return_metrics)




































