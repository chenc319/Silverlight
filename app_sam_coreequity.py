### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------- MAG7 SAM CORE EQUITY ------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
from Functions import *
from pathlib import Path
from scipy import stats
import os
DATA_DIR = os.getenv('DATA_DIR', 'data')

### SAM ###
data = '''
2025  4.63  -0.78  -3.15  0.83  4.56  2.87  ---   ---   ---   ---   ---   ---
2024  0.98  3.55  3.37  -3.74  4.04  0.02  2.43  2.21  1.53  0.30  5.52  -5.72
2023  5.51  -2.06  -1.20  -2.46  -2.93  4.99  2.58  -0.03  -2.44  -1.55  7.53  4.06
2022  -3.34  0.28  3.34  -5.60  1.41  -7.83  6.72  -1.06  -6.80  9.74  6.91  -3.64
2021  -0.92  4.57  3.46  3.48  3.20  0.73  1.67  -0.14  -2.54  3.01  -1.95  2.21
2020  -2.36  -7.47  -12.19  7.33  6.33  5.07  5.63  3.89  4.08  5.92  5.68  16.89
2019  7.35  1.92  2.36  3.57  -5.27  5.90  1.33  0.69  1.42  1.09  5.25  2.57
2018  5.94  -4.00  -0.97  -0.09  2.75  1.71  2.17  1.94  0.44  -5.65  4.32  -7.63
2017  1.69  3.76  0.07  0.24  1.13  0.93  -0.04  -0.36  2.13  1.89  3.88  0.77
2016  -1.77  2.76  6.68  -2.15  1.48  -0.02  -1.98  -1.50  -2.00  -1.98  5.23  2.22
2015  -1.98  5.94  -0.31  1.83  1.16  -1.72  -3.57  -5.68  -2.57  5.87  -1.69  -1.24
2014  -2.60  4.32  1.57  0.34  3.95  1.12  -1.75  3.19  -1.43  4.10  4.07  -0.36
2013  4.23  0.23  3.88  3.35  1.97  -0.18  4.91  -3.39  3.55  5.47  5.32  3.37
'''

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
all_dates = []
all_vals = []

for row in data.strip().split('\n'):
    parts = row.split()
    year = parts[0]
    vals = parts[1:13]
    for i, val in enumerate(vals):
        date_str = f'{year}-{months[i]}-01'  # Use the first day of each month
        if val in ['---', '--', '—']:
            decimal = float('nan')
        else:
            decimal = float(val) / 100.0
        all_dates.append(date_str)
        all_vals.append(decimal)

series = pd.Series(all_vals, index=pd.to_datetime(all_dates), name='Return')
df = series.to_frame()
df.sort_index(inplace=True)
df = df.dropna()
sam_core_equity = df.resample('ME').last()
sam_core_equity.columns = ['SAM']

### MAGS ###
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
mags_monthly_pct = each_mags_df.resample('ME').last().pct_change().dropna()

### SPX ###
with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
    spx_df = pd.read_csv(file)
    spx_df.index = pd.to_datetime(spx_df['Date'].values)
    spx_df = pd.DataFrame(spx_df['Close'])
spx_monthly_pct = spx_df.resample('ME').last().pct_change().dropna()
spx_monthly_pct.columns = ['SPX']

### MERGE ###
sam_mags_merge = merge_dfs([mags_monthly_pct, spx_monthly_pct,sam_core_equity]).dropna()
sam_mags_merge['MAGS'] = sam_mags_merge[mags_tickers].mean(axis=1)

### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------- MAG7 SAM CORE EQUITY ------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def core_equity_mags_spx():
    ### PLOTS ###
    cumulative_returns = sam_mags_merge.cumsum() * 100
    streamlit_plot(df=cumulative_returns,
                   columns_array=cumulative_returns.columns,
                   colors_array=["#7393B3", "#57666A", "#5F9EA0", "#4682B4", "#6082B6",
                                 "#849baa", "#435274", "#769A62", "#E74C3C", "#B8860B"],
                   graph_title='Summative Returns',
                   y_axis_label='%')
    streamlit_plot(df = sam_mags_merge * 100,
                   columns_array = sam_mags_merge.columns,
                   colors_array = ["#7393B3", "#57666A", "#5F9EA0", "#4682B4", "#6082B6",
                                 "#849baa", "#435274", "#769A62", "#E74C3C", "#B8860B"],
                   graph_title = 'Monthly Returns',
                   y_axis_label='%')

    ### DRAWDOWN ###
    drawdown_df = sam_mags_merge.copy()
    drawdown_df['sam_cumsum'] = sam_mags_merge['SAM'].cumsum()
    drawdown_df['spx_cumsum'] = sam_mags_merge['SPX'].cumsum()
    drawdown_df['mags_cumsum'] = sam_mags_merge['MAGS'].cumsum()
    drawdown_df['sam_drawdown'] = compute_drawdown(drawdown_df['sam_cumsum'])
    drawdown_df['spx_drawdown'] = compute_drawdown(drawdown_df['spx_cumsum'])
    drawdown_df['mag_drawdown'] = compute_drawdown(drawdown_df['mags_cumsum'])

    streamlit_drawdown_plot(df=drawdown_df,
                            graph_labels=['SAM', 'SPX', 'MAGS'],
                            df_columns_to_plot=['sam_drawdown', 'spx_drawdown', 'mag_drawdown'],
                            line_colors = ['rgba(95,179,255,1)','rgba(45,205,178,1)','rgba(13,80,185,1)'],
                            fill_colors = ['rgba(95,179,255,0.3)','rgba(45,205,178,0.3)','rgba(13,80,185,0.3)'])

def sam_core_equity_rolling_alpha():
    rolling_alpha_to_spx = pd.DataFrame(columns = ['SAM','GOOGL','AMZN','AAPL','MSFT','NVDA','TSLA'],
                                        index = sam_mags_merge.index)
    for row in range(11,len(sam_mags_merge)):
        subset = sam_mags_merge.iloc[row-11:row+1]
        sam_beta, sam_alpha, _, _, _ = stats.linregress(subset['SPX'], subset['SAM'])
        rolling_alpha_to_spx.loc[subset.index[11],'SAM'] = sam_alpha
        goog_beta, goog_alpha, _, _, _ = stats.linregress(subset['SPX'], subset['GOOGL'])
        rolling_alpha_to_spx.loc[subset.index[11],'GOOGL'] = goog_alpha
        amzn_beta, amzn_alpha, _, _, _ = stats.linregress(subset['SPX'], subset['AMZN'])
        rolling_alpha_to_spx.loc[subset.index[11],'AMZN'] = amzn_alpha
        aapl_beta, aapl_alpha, _, _, _ = stats.linregress(subset['SPX'], subset['AAPL'])
        rolling_alpha_to_spx.loc[subset.index[11],'AAPL'] = aapl_alpha
        meta_beta, meta_alpha, _, _, _ = stats.linregress(subset['SPX'], subset['META'])
        rolling_alpha_to_spx.loc[subset.index[11],'META'] = meta_alpha
        msft_beta, msft_alpha, _, _, _ = stats.linregress(subset['SPX'], subset['MSFT'])
        rolling_alpha_to_spx.loc[subset.index[11],'MSFT'] = msft_alpha
        nvda_beta, nvda_alpha, _, _, _ = stats.linregress(subset['SPX'], subset['NVDA'])
        rolling_alpha_to_spx.loc[subset.index[11],'NVDA'] = nvda_alpha
        tsla_beta, tsla_alpha, _, _, _ = stats.linregress(subset['SPX'], subset['TSLA'])
        rolling_alpha_to_spx.loc[subset.index[11],'TSLA'] = tsla_alpha
    rolling_alpha_to_spx = rolling_alpha_to_spx.dropna()
    rolling_alpha_to_spx['MAGS'] = rolling_alpha_to_spx[mags_tickers].mean(axis=1)

    ### PLOT HISTORICAL ALPHA ###
    streamlit_plot(df=rolling_alpha_to_spx * 100,
                   columns_array=['MAGS','SAM'],
                   colors_array=['#2056AE', '#E74C3C'],
                   graph_title='Rolling 12 Month Alpha',
                   y_axis_label='%')

    rolling_alpha_to_spx['spread'] = rolling_alpha_to_spx['SAM'] - rolling_alpha_to_spx['MAGS']
    streamlit_plot(df=rolling_alpha_to_spx * 100,
                   columns_array=['spread'],
                   colors_array=['#2056AE'],
                   graph_title='Alpha Spread',
                   y_axis_label='%')

    rolling_alpha_to_spx[rolling_alpha_to_spx['spread'] < 0]['MAGS'].mean(axis=0)
    rolling_alpha_to_spx[rolling_alpha_to_spx['spread'] < 0]['SAM'].mean(axis=0)
    rolling_alpha_to_spx[rolling_alpha_to_spx['spread'] > 0]['MAGS'].mean(axis=0)
    rolling_alpha_to_spx[rolling_alpha_to_spx['spread'] > 0]['SAM'].mean(axis=0)

def core_equity_mag_backtest_simulation():
    ### CALCULATE MOCK PORTFOLIOS ###
    sam_mags_merge['SAM_25'] = (sam_mags_merge['SAM'] * 0.25) + (sam_mags_merge['MAGS'] * 0.75)
    sam_mags_merge['SAM_50'] = (sam_mags_merge['SAM'] * 0.50) + (sam_mags_merge['MAGS'] * 0.50)
    sam_mags_merge['SAM_75'] = (sam_mags_merge['SAM'] * 0.75) + (sam_mags_merge['MAGS'] * 0.25)

    ### PLOTS ###
    streamlit_plot(df=sam_mags_merge.cumsum() * 100,
                   columns_array=['MAGS','SAM_25','SAM_50','SAM_75','SAM'],
                   colors_array=[
                       "#4F6D7A",  # MAGS - muted deep blue-grey
                       "#A3B18A",  # SAM_25 - soft olive green
                       "#E9C46A",  # SAM_50 - warm sand gold
                       "#E07A5F",  # SAM_75 - muted coral/orange
                       "#7EBC89",  # SAM - mellow green-teal
                       ],
                   graph_title='Mock Portfolios',
                   y_axis_label='%')
    return_metrics_df = return_metrics(sam_mags_merge[['MAGS','SAM_25','SAM_50','SAM_75','SAM']],
                                       sam_mags_merge['SPX'],
                                       12)
    streamlit_return_metrics_table(return_metrics_df)







