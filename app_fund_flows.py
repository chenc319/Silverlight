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
    df = df.resample('M').mean().dropna()
    etf_fund_flow_df = merge_dfs([etf_fund_flow_df,df])
etf_fund_flow_df = etf_fund_flow_df.ffill().dropna()
sum_etf_fund_flow_df = pd.DataFrame(etf_fund_flow_df.sum(axis=1))
sum_etf_fund_flow_df.columns = ['fund_flow']


plt.plot(etf_fund_flow_df[['spy', 'voo', 'ivv', 'qqq', 'vug', 'vea', 'acwx', 'iefa', 'vtv', 'iwf',
       'spym', 'rsp', 'schx', 'veu', 'iwb', 'smh']])
plt.show()

### ---------------------------------------------------------------------------------------------------------- ###
### --------------------------------------------- FUND FLOW MODEL -------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

fund_flow_spx_merge = merge_dfs([sum_etf_fund_flow_df,spx_monthly_pct.shift(-1)]).dropna()


