### ---------------------------------------------------------------------------------------------------------- ###
### --------------------------------------------- FUND FLOW MODEL -------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###
import pandas as pd

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
from matplotlib import pyplot as plt
DATA_DIR = os.getenv('DATA_DIR', 'data')

### ---------------------------------------------------------------------------------------------------------- ###
### --------------------------------------------- FUND FLOW MODEL -------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### EQUITIES ###
with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
    sp500 = pd.read_csv(file)
sp500.index = pd.to_datetime(sp500['Date']).values
sp500.drop('Date', axis=1, inplace=True)
spx_monthly = pd.DataFrame(sp500['Close']).resample('ME').last()
spx_monthly.columns = ['spx']
spx_monthly_pct = spx_monthly.pct_change().dropna()

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

### ---------------------------------------------------------------------------------------------------------- ###
### --------------------------------------------- FUND FLOW MODEL -------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

etf_fund_flow_std = sum_etf_fund_flow_df.rolling(12).std()
etf_fund_flow_mean = sum_etf_fund_flow_df.rolling(12).mean()
sum_etf_fund_flow_z = (sum_etf_fund_flow_df - etf_fund_flow_mean) / etf_fund_flow_std
fund_flow_spx_merge = merge_dfs([sum_etf_fund_flow_z.diff(12),spx_monthly_pct.shift(-1)]).dropna()
z_score_bucket(fund_flow_spx_merge,'fund_flow','spx')

### ---------------------------------------------------------------------------------------------------------- ###
### --------------------------------------------- FUND FLOW MODEL -------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def fund_flow_z_backtest(row,signal_colname_1):
    if row[signal_colname_1] < -0.5:
        return 1
    elif row[signal_colname_1] > 0.5:
        return 0.6
    else:
        return 1

fund_flow_spx_merge['weights'] = fund_flow_spx_merge.apply(
    lambda row: fund_flow_z_backtest(row,
                                           'fund_flow'), axis=1
)
fund_flow_spx_merge['bt_returns'] = fund_flow_spx_merge['weights'] * fund_flow_spx_merge['spx']

spx_fund_flow_return_metrics = return_metrics(
    fund_flow_spx_merge[['bt_returns','spx']],
    fund_flow_spx_merge[['spx']],
    12
)
spx_fund_flow_return_metrics['Return/Risk']

### ---------------------------------------------------------------------------------------------------------- ###
### --------------------------------------------- FUND FLOW MODEL -------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

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
                   y_axis_label = 'Rolling 12m Change')
    streamlit_plot(df=sum_etf_fund_flow_df,
                   columns_array=etf_fund_flow_df.columns,
                   colors_array=[
                       "#000000",  # black
                   ],
                   graph_title='Top 6 Net Fund Flow',
                   y_axis_label='Rolling 12m Change')

def equity_fund_flow_results():
    streamlit_plot(df=(1 + fund_flow_spx_merge[['bt_returns', 'spx']]).cumprod() - 1,
                   columns_array=['bt_returns', 'spx'],
                   colors_array=["#8B0000", "#000000"],
                   graph_title='Equities Historical Performance',
                   y_axis_label='%')
    streamlit_return_metrics_table(spx_fund_flow_return_metrics)
