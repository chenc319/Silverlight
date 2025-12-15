### --------------------------------------------------------------------------------------- ###
### ----------------------------------- FUND FLOW MODEL ----------------------------------- ###
### --------------------------------------------------------------------------------------- ###

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
from matplotlib import pyplot as plt
DATA_DIR = os.getenv('DATA_DIR', 'data')

### --------------------------------------------------------------------------------------- ###
### ----------------------------------- FUND FLOW MODEL ----------------------------------- ###
### --------------------------------------------------------------------------------------- ###

### EQUITIES ###
with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
    sp500 = pd.read_csv(file)
sp500.index = pd.to_datetime(sp500['Date']).values
sp500.drop('Date', axis=1, inplace=True)
spx_monthly = pd.DataFrame(sp500['Close']).resample('ME').last()
spx_monthly.columns = ['spx']
spx_monthly_pct = spx_monthly.pct_change().dropna()


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

### FUND FLOWS ###
with open(Path(DATA_DIR) / 'ETF Fund Flows.xlsx', 'rb') as file:
    etf_flows = pd.read_excel(file,sheet_name='clean')
etf_fund_flow_df = pd.DataFrame()
for x in range(0,len(etf_flows.columns)-1,2):
    df = etf_flows.iloc[:,[x,x+1]]
    df.index = pd.to_datetime(df.iloc[:,0].values)
    df.drop(df.columns[0], axis=1, inplace=True)
    df = df[df != 0]
    df = df.resample('ME').mean().dropna()
    etf_fund_flow_df = merge_dfs([etf_fund_flow_df,df])
etf_fund_flow_df = etf_fund_flow_df.ffill().dropna()
etf_fund_flow_df = etf_fund_flow_df[['voo','vug','vtv','ivv','spy','vea']]
sum_etf_fund_flow_df = pd.DataFrame(etf_fund_flow_df.sum(axis=1))
sum_etf_fund_flow_df.columns = ['fund_flow']

### --------------------------------------------------------------------------------------- ###
### ----------------------------------- FUND FLOW MODEL ----------------------------------- ###
### --------------------------------------------------------------------------------------- ###

### SPX ###
etf_fund_flow_std = sum_etf_fund_flow_df.rolling(12).std()
etf_fund_flow_mean = sum_etf_fund_flow_df.rolling(12).mean()
sum_etf_fund_flow_z = (sum_etf_fund_flow_df - etf_fund_flow_mean) / etf_fund_flow_std
fund_flow_spx_merge = merge_dfs([sum_etf_fund_flow_df,
                                 sum_etf_fund_flow_z.diff(12),
                                 spx_monthly_pct.shift(-1)]).dropna()
fund_flow_spx_merge.columns = ['fund_flow','fund_flow_z','spx']
z_score_bucket(fund_flow_spx_merge,'fund_flow_z','spx')

### MAG7 ###
fund_flow_mags_merge = merge_dfs([sum_etf_fund_flow_df,
                                 sum_etf_fund_flow_z,
                                 mock_mags_monthly_pct.shift(-1)]).dropna()
fund_flow_mags_merge.columns = ['fund_flow','fund_flow_z','mags']
z_score_bucket(fund_flow_mags_merge,'fund_flow_z','mags')

### --------------------------------------------------------------------------------------- ###
### ----------------------------------- FUND FLOW MODEL ----------------------------------- ###
### --------------------------------------------------------------------------------------- ###

def fund_flow_spx_z_backtest(row,signal_colname_1):
    if row[signal_colname_1] < -0.5:
        return 1
    elif row[signal_colname_1] > 0.5:
        return 0.6
    else:
        return 1

fund_flow_spx_merge['z_weights'] = fund_flow_spx_merge.apply(
    lambda row: fund_flow_spx_z_backtest(row,
                                           'fund_flow_z'), axis=1
)

fund_flow_spx_merge['bt_returns'] = (fund_flow_spx_merge['z_weights'] *
                                     fund_flow_spx_merge['spx'])
spx_fund_flow_return_metrics = return_metrics(
    fund_flow_spx_merge[['bt_returns','spx']],
    fund_flow_spx_merge[['spx']],
    12
)
spx_fund_flow_return_metrics['Return/Risk']

def fund_flow_mags_z_backtest(row,signal_colname_1):
    if row[signal_colname_1] < -1:
        return 0.6
    elif row[signal_colname_1] > -1:
        return 1
    else:
        return 1

fund_flow_mags_merge['z_weights'] = fund_flow_mags_merge.apply(
    lambda row: fund_flow_mags_z_backtest(row,
                                           'fund_flow_z'), axis=1
)

fund_flow_mags_merge['bt_returns'] = (fund_flow_mags_merge['z_weights'] *
                                     fund_flow_mags_merge['mags'])
mags_fund_flow_return_metrics = return_metrics(
    fund_flow_mags_merge[['bt_returns','mags']],
    fund_flow_mags_merge[['mags']],
    12
)
mags_fund_flow_return_metrics['Return/Risk']

### --------------------------------------------------------------------------------------- ###
### ----------------------------------- FUND FLOW MODEL ----------------------------------- ###
### --------------------------------------------------------------------------------------- ###

def display_etf_fund_flows():
    streamlit_plot(df = etf_fund_flow_df,
                   columns_array = etf_fund_flow_df.columns,
                   colors_array = [
                       "#6BA368",  # soft green
                       "#C75C5C",  # muted red
                       "#D4AF37",  # muted golden yellow
                       "#E29547",  # soft orange
                       "#4F81BD",  # muted blue
                       "#B38CB4",  # muted purple
                       ],
                   graph_title = 'Top 6 Fund Flows',
                   y_axis_label = 'Rolling 12m Change (Millions)')
    streamlit_plot(df=sum_etf_fund_flow_df,
                   columns_array=sum_etf_fund_flow_df.columns,
                   colors_array=[
                       "#000000",  # black
                   ],
                   graph_title='Top 6 Net Fund Flow',
                   y_axis_label='Rolling 12m Change (Millions)')

def equity_fund_flow_results():
    streamlit_plot(df=(1 + fund_flow_spx_merge[['bt_returns',
                                                'spx']]
                       ).cumprod() - 1,
                   columns_array=['bt_returns', 'spx'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='Equities Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(spx_fund_flow_return_metrics)

def mags_fund_flow_results():
    streamlit_plot(df=(1 + fund_flow_mags_merge[['bt_returns',
                                                'mags']]
                       ).cumprod() - 1,
                   columns_array=['bt_returns', 'mags'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='Equities Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(mags_fund_flow_return_metrics)
