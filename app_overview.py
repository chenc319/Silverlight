### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME OVERVIEW --------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
DATA_DIR = os.getenv('DATA_DIR', 'data')

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME OVERVIEW --------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### EQUITIES ###
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

### BONDS ###
with open(Path(DATA_DIR) / 'AGG.csv', 'rb') as file:
    agg = pd.read_csv(file)
agg.index = pd.to_datetime(agg['Date']).values
agg.drop('Date', axis=1, inplace=True)
bonds_weekly = pd.DataFrame(agg['Close']).resample('W-FRI').last()
bonds_weekly.columns = ['bonds']
bonds_daily = pd.DataFrame(agg['Close'])
bonds_daily.columns = ['bonds']
bonds_monthly = pd.DataFrame(agg['Close']).resample('ME').last()
bonds_monthly.columns = ['bonds']

### YIELDS ###
with open(Path(DATA_DIR) / 'treasury_1m.pkl', 'rb') as file:
    treasury_1m = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'treasury_2y.pkl', 'rb') as file:
    treasury_2y = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'treasury_5y.pkl', 'rb') as file:
    treasury_5y = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'treasury_10y.pkl', 'rb') as file:
    treasury_10y = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'treasury_30y.pkl', 'rb') as file:
    treasury_30y = pd.read_pickle(file)

treasury_merge = merge_dfs([treasury_1m,treasury_2y, treasury_5y, treasury_10y,treasury_30y]).dropna()
treasury_merge.index = pd.to_datetime(treasury_merge.index).values
treasury_merge.columns = ['1m','2y','5y','10y','30y']
treasury_monthly_df = treasury_merge.resample('ME').last()


### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME OVERVIEW --------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_grid_nowcast():
    print('hi')

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME OVERVIEW --------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_flowcluster_nowcast():
    print('hi')


### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME OVERVIEW --------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_crossasset_nowcast():
    print('hi')


### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME OVERVIEW --------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_yc_nowcast():
    print('hi')




