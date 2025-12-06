### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------- 401K SEASONALITY MODEL ----------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
DATA_DIR = os.getenv('DATA_DIR', 'data')

### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------- 401K SEASONALITY MODEL ----------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### EQUITIES ###
with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
    sp500 = pd.read_csv(file)
sp500.index = pd.to_datetime(sp500['Date']).values
sp500.drop('Date', axis=1, inplace=True)
spx_monthly = pd.DataFrame(sp500['Close']).resample('ME').last()
spx_monthly.columns = ['spx']

### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------- 401K SEASONALITY MODEL ----------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### BONDS ###
with open(Path(DATA_DIR) / 'AGG.csv', 'rb') as file:
    agg = pd.read_csv(file)
agg.index = pd.to_datetime(agg['Date']).values
agg.drop('Date', axis=1, inplace=True)
bonds_monthly = pd.DataFrame(agg['Close']).resample('ME').last()
bonds_monthly.columns = ['bonds']

### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------- 401K SEASONALITY MODEL ----------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### MERGE DFS AND PREPARE ###
spx_bonds_merge = merge_dfs([spx_monthly,bonds_monthly]).pct_change().dropna()
spx_bonds_merge_mean = spx_bonds_merge.rolling(12).mean()
spx_bonds_merge_std = spx_bonds_merge.rolling(12).std()
spx_bonds_merge_z = ((spx_bonds_merge - spx_bonds_merge_mean) / spx_bonds_merge_std).dropna()
spx_bonds_merge_z['spread'] = spx_bonds_merge_z['spx'] - spx_bonds_merge_z['bonds']

spx_bonds_merge_z['month_num'] = spx_bonds_merge_z.index.month
spx_bonds_merge_z['month_name'] = spx_bonds_merge_z.index.month_name()

# group by month number, compute average spread
seasonality = (
    spx_bonds_merge_z.groupby('month_num')['spread']
      .mean()
      .to_frame(name='avg_spread')
)

# optional: attach month names for readability
seasonality['month'] = seasonality.index.map(lambda m: pd.Timestamp(2000, m, 1).strftime('%b'))

# final seasonality DataFrame with Jan..Dec in order
seasonality = seasonality[['month', 'avg_spread']]
print(seasonality)




